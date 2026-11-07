# Release Notes — Tab Agent Pro 1.1.0

**Release date:** 2026-11-07
**Package:** `artifacts/phase6_release_package.zip` (58 files, 600 KiB, CRC-verified)
**Headline:** every advertised entry point works as documented, and the service can
now tell you when it is broken instead of reporting `up`.

---

## What this release is

A production-readiness pass over the whole repository: audit, claim verification,
build validation, dynamic testing, security hardening, documentation, and
sign-off. No API changes. No breaking changes to the CLI, the web UI, or the
REAPER scripts. Everything previously working still works the same way; what
changes is that the failures are now visible and the limits are enforced.

## Highlights

### 1. Health is real, and readiness is separate
`/health` used to say `up` even with every transcription backend dead — it could
not detect the exact failure it existed to detect. Now:

| Endpoint | Semantics |
|---|---|
| `/health` | liveness + per-component detail, always `200` while the process lives |
| `/ready` | `200` only when a required backend is loadable, else `503` naming the blocker |
| `/health/metrics` | bounded per-stage timings and error rates |

The container `HEALTHCHECK` probes `/ready`, so a broken image is taken out of
rotation instead of serving a dead UI.

### 2. Concurrent requests no longer collide
Every request gets its own `mkdtemp` session. Previously two simultaneous users
shared one output directory and could receive each other's stems and ZIPs.

### 3. Ten silent failure sites are now observable
`.env` parsing, corrupt `user_preferences.json`, YourMT3+ clone timeout, clone
non-zero exit, malformed token streams, and empty audio-read messages all used
to `pass` or `return None`. Each now logs structured context, updates component
health, and either recovers with a documented fallback or escalates.

### 4. Supply chain is pinned and gated
- `requirements.lock` — 118 exact pins covering the runtime closure of what actually ships (CPU-only: no TensorFlow, no CUDA wheels).
- `tools/check_lockfile.py` — fails CI and the Docker build if the installed
  environment drifts (and `--static` catches a dependency added without a pin).
- `tools/check_vulns.py` — audits the installed set and fails on any advisory
  without an accepted, **unexpired** entry in `.pip-audit-ignore`.
- The YourMT3+ checkpoint can be revision-pinned, and
  `YOURMT3_REQUIRE_PINNED_REVISION=1` makes an unpinned download a hard error.

### 5. Container hardening
Non-root `tabagent` user, application code read-only at runtime, only cache and
output directories writable, CPU-only torch, exec-form `CMD`, `/ready`
healthcheck, lockfile gate enforced during the build.

### 6. Docs that match the code
- README: tablature sample regenerated from actual output; 50 MB / 5 min scoped
  to the web UI; the GPU claim made conditional; env vars, test counts, and the
  deploy file list brought current.
- New: `docs/RUNBOOK.md`, `docs/ACCEPTED_RISKS.md`, `SECURITY.md`,
  `THIRD_PARTY_NOTICES.md`, and `AUDIT.md` Appendix C (post-fix verification).

## Upgrade notes

- **No migration.** There is no persistent state to migrate; the model cache is
  a cache and can be deleted.
- **New optional env vars:** `YOURMT3_REVISION`, `YOURMT3_REQUIRE_PINNED_REVISION`
  (recommended `1` in production), `MAX_CONCURRENT_REQUESTS`, `QUEUE_MAX_SIZE`,
  `UPLOAD_TTL_SEC`. Defaults are safe; `.env.example` documents all 11.
- **`setuptools<81` stays.** Do not apply pip-audit's suggested upgrade: 81+
  removes `pkg_resources`, which `pretty_midi` imports (reproduced failure in
  `docs/ACCEPTED_RISKS.md` §R-2).
- **Python 3.11+** (was 3.10+). The pinned set requires it — 16 pins declare
  `requires-python >= 3.11` — and CI now runs 3.12 to match the image. Basic
  Pitch still installs `--no-deps` on Linux;
  the Dockerfile and CI both encode the two-stage install.

## Verification at release

| Gate | Result |
|---|---|
| `ruff check .` | pass |
| `black --check .` | 31 files unchanged |
| `mypy .` | no issues in 31 source files |
| `pytest` | **290 passed**, 91% coverage |
| `tools/simulate_failures.py` | 15/15 |
| `validate.py --quick` | 101 pass / 0 fail / 1 skip |
| `bandit -r .` | 0 issues |
| `tools/check_vulns.py` | 0 unaccepted (70 accepted, expiring 2027-01-31) |
| `tools/check_index.py` | ReaPack-conformant, commit resolves |
| `tools/check_env_example.py` | 11/11 variables referenced |
| `tools/check_docker_install.py` | pass (incl. hardening guards) |
| `tools/check_lockfile.py` | 118 pins verified; 19 declared deps covered |

## Known limitations (stated, not hidden)

- **No Docker daemon in the audit environment** → the image build, staging
  deploy, and rollback drill are validated by static guards plus running every
  command outside the container. Not a real build.
- **YourMT3+ happy path is unverified** here (gated multi-GB checkpoint, no Hub
  egress). It is instrumented, pinned, and fail-closed.
- **Demucs needs its weights** for stem separation; without them audio passes
  through unseparated with a logged warning rather than failing.
- **No authentication or per-IP rate limiting** — by design for a single-tenant
  Space; the queue is bounded so abuse degrades to rejection (see
  `docs/ACCEPTED_RISKS.md` R-5/R-6).

## Deploy

```bash
export HF_TOKEN=hf_...
./tools/deploy_space.sh <owner>/<space>
curl -s -o /dev/null -w '%{http_code}\n' https://<host>/ready   # expect 200
```

Rollback is a re-deploy of the previous revision; the procedure is in
`docs/RUNBOOK.md` §4.
