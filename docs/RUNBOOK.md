# Operations Runbook

Everything an on-call engineer needs to run, verify, roll back, and recover Tab
Agent Pro. Written so it can be followed at 3 a.m. by someone who did not build
the service.

- **Service:** Gradio web UI + FastAPI health endpoints (`7860`)
- **Shape:** single container, CPU-bound, stateless apart from the model cache
  and a short-lived upload/temp directory
- **Failure philosophy:** the app never reports `up` for a component it could not
  load. If transcription is gone, `/ready` is `503` and the platform takes it
  out of rotation.

---

## 1. Deploy

```bash
# 1. Verify the tree is release-clean (same gates CI runs)
ruff check . && black --check . && mypy .
python -m pytest tests/ -q
python tools/check_vulns.py
python tools/check_lockfile.py

# 2. Ship it (pinned files only; the Space gets exactly what the image needs)
export HF_TOKEN=hf_...
./tools/deploy_space.sh <owner>/<space>
```

The script uploads `Dockerfile`, the two requirement files, the app modules,
`README.md` (it carries the Space YAML frontmatter) and `examples/`, then prints
the Space URL. There is no separate "build" step: the Space builds the image.

**Production environment variables** (Space settings → Variables and secrets):

| Name | Value | Why |
|---|---|---|
| `YOURMT3_REQUIRE_PINNED_REVISION` | `1` | Fail closed if the checkpoint revision is not pinned |
| `YOURMT3_REVISION` | reviewed commit sha | Removes the `main`-branch supply-chain vector |
| `HF_TOKEN` | **secret**, only if the checkpoint repo is gated | Never a build arg |
| `MAX_CONCURRENT_REQUESTS` | `1` | One transcription at a time per CPU worker |
| `QUEUE_MAX_SIZE` | `32` | Bounded backlog; beyond this requests are rejected |

---

## 2. Health and readiness

```bash
curl -s https://<host>/health | jq .
curl -s -o /dev/null -w '%{http_code}\n' https://<host>/ready   # 200 = serving
curl -s https://<host>/health/metrics | jq .
```

- `/health` — always `200` while the process is alive. It reports overall status
  plus per-component state (`transcription`, `demucs`, `gradio`, ...). Use it to
  answer *"what is broken?"*.
- `/ready` — `200` when a usable **transcription** backend is loaded; `503`
  otherwise, with the blocking component in the body. Use it for traffic gating
  and as the container healthcheck.

  Readiness is gated on transcription only. **Demucs is deliberately not a
  readiness gate**: separation has a documented raw-audio passthrough, so an
  instance without Demucs weights still transcribes whole-mix audio (with the
  quality loss reported in `degraded_components`). Gating on it made a perfectly
  serving instance answer `503` — which would pull it out of rotation for no
  reason. Watch `degraded_components` for quality, `/ready` for availability.
- `/health/metrics` — per-stage `count`, `total_s`, `avg_s`, `error_rate` plus a
  bounded window average.

**Status semantics**

| Status | Meaning |
|---|---|
| `up` | Component loaded and the last operation succeeded |
| `degraded` | Loaded, but with recoverable errors (e.g. Demucs API failed, CLI fallback working) |
| `down` | Not loaded or the last operation failed unrecoverably |
| `error: <reason>` | Load or operation failed with a reason — read the JSON logs for the full detail |

**Alerts worth wiring**

| Signal | Threshold | Meaning |
|---|---|---|
| `/ready` != 200 for 2 min | page | Nothing can transcribe |
| `components.basic_pitch` not `available`/`loaded` | page | Basic Pitch gone (bad image, missing ONNX runtime) |
| `degraded_components` contains `demucs` for > 30 min | ticket | Stem separation is falling back to passthrough — output degrades to whole-mix transcription |
| `metrics.process_audio.error_rate` > 0.25 over 15 min | ticket | Users are hitting failures |
| `avg_s` for `basic_pitch_transcription` > 3× baseline (`0.82 s` for a 2 s clip) | ticket | CPU starvation or model reloading every request |
| container RSS > 1.5 GB | ticket | Unexpected growth; see §7 |

---

## 3. Logs

Single-line JSON on stdout — no external log service required, so they survive
platform restarts and work on HF Spaces.

```json
{"ts":"2026-11-07T05:20:20.021Z","level":"METRIC","module":"metrics",
 "event":"stage.basic_pitch_transcription.elapsed_s","uptime_s":49.5,
 "value":0.13,"unit":"s","success":true,"error_rate":0.0}
```

Useful queries (with `jq`):

```bash
# Every error, with the component that reported it
jq -c 'select(.level=="ERROR") | {ts,module,event,error}' app.log

# Follow one user's failure by the reference shown in the UI
jq -c 'select(.error_ref=="<ref from the UI message>")' app.log

# Failures by event name over a day
jq -r 'select(.level=="ERROR") | .event' app.log | sort | uniq -c | sort -rn
```

**Adding an error tracker (Sentry/Rollbar):** the hook point is
`monitoring.get_logger()` — records already carry `level`, `module`, `event`,
and exception text. Pipe `level=="ERROR"` records to the tracker; no code change
to the pipeline is needed. This is deliberately not a default dependency.

---

## 4. Rollback

The Space keeps revision history, so a rollback is a re-deploy of a known-good
revision.

```bash
# 1. Find the last good revision
git log --oneline -10
huggingface-cli repo-files <owner>/<space> --repo-type space   # or the Space UI "History"

# 2. Re-deploy that revision's tree
git checkout <good-sha>
./tools/deploy_space.sh <owner>/<space>

# 3. Verify (the readiness probe IS the go/no-go signal)
curl -s -o /dev/null -w '%{http_code}\n' https://<host>/ready   # expect 200
curl -s https://<host>/health | jq -r '.status'                  # expect "up"
```

> **Rolling back past 1.1.0:** releases before `1.1.0` have no `/ready` route, so
> it answers `404`. Treat that as *"this target predates the endpoint"*, not as a
> failed rollback — `/health` returning `200` plus one successful transcribe is
> the go/no-go signal in that case. (Verified against `e7cd693` during sign-off:
> `404` on `/ready`, `200` on `/health`, and `python main.py` still produced all
> twelve output files.) Prefer rolling back to a release that has the endpoint
> when one exists.

If the rollback target is also broken, roll back further — do not attempt a
forward fix under pressure. The pipeline has no schema or persistent state, so a
rollback is always safe and never needs a data repair.

**Rollback rehearsal (do this quarterly, it takes two minutes):** deploy the
current revision to a staging Space, re-deploy the previous revision, confirm
`/ready` 200s and a single end-to-end transcribe. Recorded in the release
checklist.

---

## 5. Dependency refresh (forced every 12 months)

`.pip-audit-ignore` entries expire **2027-01-31**; on that date `check_vulns.py`
fails and no release can ship until each advisory is re-reviewed.

```bash
# 1. What is failing and why
python tools/check_vulns.py

# 2. Fix or re-justify, then re-scan
#    - fix: bump the pin in requirements.txt, regenerate the lock, re-run everything
python tools/check_lockfile.py --write
python -m pip_audit --format json > /tmp/audit.json

# 3. Re-lock, re-test, re-measure
python -m pytest tests/ -q
python /home/user/tab-agent-pro/tools/simulate_failures.py
```

Rules that are not negotiable:

- **`setuptools<81` stays.** 81+ removed `pkg_resources`, which `pretty_midi`
  imports (verified: `setuptools==84.1.0` → `ModuleNotFoundError` → no
  transcription). Do not apply pip-audit's advertised fix.
- **Never accept an advisory without a reason and a date** — the gate rejects
  entries with a missing or past expiry.
- **License check after every bump:** anything GPL/AGPL/SSPL is a release
  blocker (`THIRD_PARTY_NOTICES.md`).

---

## 6. Capacity and cost

- One request is CPU-bound for tens of seconds: Demucs separation dominates when
  the weights are present; Basic Pitch adds ~0.8 s per 2 s of audio per stem.
- Keep `MAX_CONCURRENT_REQUESTS=1` on a 2-vCPU box. Raising it helps throughput
  only with more cores; memory is ~600 MB resident after imports plus ~90 MB per
  in-flight transcription.
- Uploads are capped at 50 MB / 5 minutes, so a single request cannot exceed the
  memory budget.
- Disk: session dirs are pruned after 1 h, uploads after `UPLOAD_TTL_SEC`

(900 s). Both pruners run on each request, so a busy instance self-cleans; an
  idle instance cleans up on the next request, not on a timer.

---

## 7. Known failure modes

| Symptom | Cause | Action |
|---|---|---|
| `/ready` 503, `/health` shows `transcription: down` | Basic Pitch/ONNX missing from the image | Check the build logs for the `--no-deps` basic-pitch install and the `mir_eval`/`resampy` pins; `python tools/check_docker_install.py` reproduces it locally |
| `/health` `status: degraded` with `degraded_components: ["demucs"]`, logs say `demucs_api_failed` | Demucs API error, CLI fallback in use | Non-fatal; separation still happens, just slower. If `demucs_all_failed` also appears, audio passes through unseparated and tab quality drops |
| Transcription succeeds but the tab is sparse | Notes filtered as out-of-range for the instrument | Expected: `last_skipped_notes` and a UI warning name the count. Check the instrument/profile |
| First request after a deploy takes ~60 s | torch/ONNX session cold start | Expected once per container. Do not put it in the alert window without a warm-up |
| RSS grows on every request | Should not happen — measured 0.1 MB/run steady state | If it does, capture `/health/metrics`, then restart the container (state is disposable) and file an issue with the numbers |
| Requests rejected with a queue message | `QUEUE_MAX_SIZE` reached | Capacity issue, not a bug. Scale cores or raise the limit deliberately |
| ZIP is empty or short of tracks | A stem had no playable notes | Read the warning text in the UI; it names which track and how many notes were skipped |

---

## 8. Release checklist

- [ ] All gates green: `ruff`, `black --check`, `mypy`, `pytest` (290+),
      `simulate_failures.py` (15/15), `check_index.py`, `check_env_example.py`,
      `check_docker_install.py`, `check_lockfile.py` (+ `--static`),
      `check_vulns.py`, `validate.py`
- [ ] `requirements.lock` regenerated if any dependency changed
- [ ] `.pip-audit-ignore` reviewed; no expiry in the past
- [ ] `THIRD_PARTY_NOTICES.md` updated if dependencies were added/removed
- [ ] `CHANGELOG.md` has an entry for the version being shipped
- [ ] README claims re-read against the code (test counts, limits, GPU wording)
- [ ] Deployed, `/ready` 200, `/health` `up`, one end-to-end transcribe by hand
- [ ] Rollback revision recorded in the release notes
