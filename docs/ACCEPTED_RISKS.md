# Accepted Risks

Every entry below is a risk the project **knowingly** carries. CI enforces this
list: `tools/check_vulns.py` fails the build on any advisory that is not listed
here with an unexpired date, and fails again once an entry expires. Adding an
entry is a reviewable act; leaving one to rot is not possible.

- **Ignore file:** `.pip-audit-ignore` (machine-readable, `<ID> <expiry> <reason>`)
- **All entries expire:** `2027-01-31` — on that date CI turns red until each
  one is re-reviewed and either re-justified or fixed.
- **Scope of the scan:** the *installed* environment, i.e. exactly what ships
  (`requirements.lock`, 164 distributions), not the looser `requirements.txt`.
- **Baseline at sign-off:** 70 advisories across 4 packages, 0 unaccepted.

---

## R-1 — `torch` 2.5.1 (23 advisories)

**Why it is here:** torch is the largest dependency and carries the most
advisories in any given release. The advisories cluster around weight loading,
`torch.jit`/TorchScript, `torch.distributed`, and pickle deserialisation of
untrusted model files.

**Exposure in this project:** low.

- Inference only. No training loop, no gradient storage of user data.
- The **only** `torch.load` call reads the YourMT3+ checkpoint with
  `weights_only=True` (grep-verified single call site in `agents.py`), so no
  arbitrary pickle objects can be instantiated.
- No user-supplied model files are ever accepted. There is no upload path for
  weights, and no path is built from user input.
- `torch.jit`, `torch.compile`, `torch.inductor` and `torch.distributed` are not
  imported anywhere in the tree.
- The checkpoint source is revision-pinnable (`YOURMT3_REVISION`) and can be made
  to fail closed (`YOURMT3_REQUIRE_PINNED_REVISION=1`), which removes the
  remaining supply-chain vector.

**Why not upgrade now:** moving torch means re-validating the ONNX/Basic Pitch
and Demucs paths on CPU and GPU, re-measuring the performance baseline, and
re-testing the 290-test suite — a planned dependency-refresh task, not a hotfix.
The upgrade is tracked in the release checklist.

**Revisit by:** 2027-01-31 (hard CI failure).

## R-2 — `setuptools` `<81` (7 advisories)

**Why it is here:** `pretty_midi` imports `pkg_resources`. `setuptools` 81
removed it, and 83/84 do not ship it at all.

**Proof the pin is load-bearing** (reproduced in an isolated venv, not asserted
from memory): with `setuptools==84.1.0`, `import pretty_midi` raises
`ModuleNotFoundError: No module named 'pkg_resources'`, which breaks Basic Pitch
transcription — the always-available fallback — and therefore the product.

**Exposure:** install-time only. The affected code paths are
`package_index`/sdist fetching in the build tool. Runtime never imports
`setuptools`.

**Decision:** the pin stays. **Do not apply pip-audit's advertised fix.** If
`pretty_midi` ever ships a `pkg_resources`-free release, revisit.

**Revisit by:** 2027-01-31, or when `pretty_midi` drops `pkg_resources`.

## R-3 — `pillow` (39 advisories)

**Why it is here:** `pillow` is not a declared dependency of this project and is
never imported (`grep -rn "import PIL"` → no matches). It enters the environment
only through the TensorFlow chain that `basic-pitch` *declares* but which this
project deliberately does not install (the ONNX backend needs no TensorFlow, and
TensorFlow has no Python 3.12+ wheels below 2.15.1).

**Exposure:** none in the shipped configuration. It is listed so that (a) the
scan's output is fully accounted for, and (b) if a future feature adds image
handling, the decision is explicit rather than inherited silently.

**Decision:** accepted, not fixed. If TF ever becomes a runtime dependency, this
becomes a real risk and must be re-evaluated.

**Revisit by:** 2027-01-31.

## R-4 — `black` (1 advisory)

**Why it is here:** dev-only formatter, not installed in the runtime image.

**Exposure:** none in production.

**Revisit by:** 2027-01-31.

---

## Non-CVE accepted risks

These are recorded for completeness; they are design decisions, not oversights.

| # | Risk | Why accepted | Mitigation / revisit trigger |
|---|---|---|---|
| R-5 | **No authentication** | The product is a single-tenant demo/Space; there is nothing to authenticate against and no per-user data to protect. | If it becomes multi-tenant or handles private audio, add auth at the proxy. |
| R-6 | **No per-IP rate limiting** | The Gradio queue is bounded (`MAX_CONCURRENT_REQUESTS`, `QUEUE_MAX_SIZE`), so abuse degrades to fast rejection rather than memory exhaustion. | Add a proxy-level limiter for any public deployment. |
| R-7 | **`soxr` is LGPL-2.1-or-later** | Dynamically linked, unmodified, replaceable wheel; LGPL permits this in an MIT project. | Documented in `THIRD_PARTY_NOTICES.md`. |
| R-8 | **Optional YourMT3+ checkpoint has upstream-defined terms and a `torch.load` path** | It is optional (Basic Pitch is the default working path), and it is pinned + fail-closed capable. | Run production with `YOURMT3_REQUIRE_PINNED_REVISION=1`. |
| R-9 | **No external error tracker (Sentry etc.)** | Keeps the image small and dependency-light; JSON logs already carry an `error_ref` for correlation. | Hook point documented in `docs/RUNBOOK.md`. |
| R-10 | **Uploads are written to disk before validation completes** | Validation happens before decode; the file must exist to be probed, and the TTL pruner bounds disk use. | `UPLOAD_TTL_SEC` (default 900 s) plus per-request session cleanup. |

---

## How to re-review

```bash
# 1. What is currently accepted, and when does it expire?
cat .pip-audit-ignore

# 2. What does the environment actually contain?
python tools/check_vulns.py            # exits 1 on anything unaccepted/expired

# 3. Is an advisory still real for us? Check the call sites:
grep -rn "torch.load\|import PIL\|torch.jit\|torch.compile" --include=*.py .

# 4. After fixing or re-justifying, update the expiry and re-run.
```
