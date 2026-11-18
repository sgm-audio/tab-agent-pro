# Accepted Risks

`.pip-audit-ignore` is the machine-readable list (`<ID> <expiry> <reason>`).
A reported advisory fails CI unless it has an unexpired entry. The current
entries expire on `2027-01-31`. The scan covers the installed environment, so
a run in the CI virtualenv can report issues in dev tooling too. It is not an
attestation of an image that was not built and scanned.

---

## R-1 — `torch` 2.5.1

**Why it is here:** torch is the largest dependency and carries the most
advisories in any given release. The advisories cluster around weight loading,
`torch.jit`/TorchScript, `torch.distributed`, and pickle deserialisation of
untrusted model files.

**Exposure in this project:** low.

- Inference only. No training loop, no gradient storage of user data.
- No user-supplied model files are ever accepted. There is no upload path for
  weights, and no path is built from user input.
- `torch.jit`, `torch.compile`, `torch.inductor` and `torch.distributed` are not
  imported anywhere in the tree.
- Basic Pitch ships its ONNX weights in the wheel. Demucs may download and
  deserialize upstream checkpoints on first use. Locking the Python package
  does not independently pin each remote model file; upstream model trust is
  still required.

**Why not upgrade now:** moving torch means re-validating the ONNX/Basic Pitch
and Demucs paths on CPU and GPU, re-measuring the performance baseline, and
re-running the test suite — a planned dependency-refresh task, not a hotfix.
The upgrade is tracked in the release checklist.

**Revisit by:** 2027-01-31 (hard CI failure).

## R-2 — `setuptools` 80.10.2

`pretty-midi==0.2.11.post0` no longer imports `pkg_resources`, but Basic
Pitch 0.4.0 requires `resampy<0.4.3`, which still does. Basic Pitch fails to
import with setuptools 84.2.1. The compatible version is now pinned in the
runtime lock. The accepted advisories refer to installation and package
building, which never run when the service handles an audio request. Recheck
when Basic Pitch supports a resampy release that no longer needs this API.

## R-3 — retired Pillow exceptions

Pillow is installed via Gradio. The lock moves to Gradio 6.32 and Pillow
13.0.0 to address the audited Pillow findings, and to Tornado 6.6.1 to
address its reported finding. The Pillow exceptions have been removed.

## R-4 — `black` (1 advisory)

**Why it is here:** dev-only formatter, not installed in the runtime image.

**Exposure:** none in production.

**Revisit by:** 2027-01-31.

---

## Non-CVE accepted risks

These are recorded for completeness; they are design decisions, not oversights.

| # | Risk | Why accepted | Mitigation / revisit trigger |
|---|---|---|---|
| R-5 | **No authentication** | The product is a single-tenant tool; there is nothing to authenticate against and no per-user data to protect. | If it becomes multi-tenant or handles private audio, add auth at the proxy. |
| R-6 | **No per-IP rate limiting** | The Gradio queue is bounded (`MAX_CONCURRENT_REQUESTS`, `QUEUE_MAX_SIZE`), so abuse degrades to fast rejection rather than memory exhaustion. | Add a proxy-level limiter for any public deployment. |
| R-7 | **`soxr` is LGPL-2.1-or-later** | Dynamically linked, unmodified, replaceable wheel; LGPL permits this in an MIT project. | Documented in `THIRD_PARTY_NOTICES.md`. |
| R-8 | **No external error tracker (Sentry etc.)** | Keeps the image small and dependency-light; JSON logs already carry an `error_ref` for correlation. | Hook point documented in `docs/RUNBOOK.md`. |
| R-9 | **Uploads are written to disk before validation completes** | Validation happens before decode; the file must exist to be probed, and the TTL pruner bounds disk use. | `UPLOAD_TTL_SEC` (default 900 s) plus per-request session cleanup. |

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
