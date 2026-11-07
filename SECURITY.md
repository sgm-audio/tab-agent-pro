# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 1.1.x | ✅ |
| 1.0.x | ⚠️ security fixes only |
| < 1.0 | ❌ |

## Reporting a vulnerability

Please **do not** open a public issue for a security problem.

- Open a private report at <https://github.com/sgm-audio/tab-agent-pro/security/advisories/new>
- Or email the maintainer listed in `pyproject.toml`
- Include: affected version/commit, reproduction steps, impact, and any suggested fix.

You can expect an acknowledgement within 3 business days and a remediation plan
within 10. Credit is given in `CHANGELOG.md` unless you ask to stay anonymous.

## Scope and threat model

Tab Agent Pro is a **single-tenant** audio transcription service. It has **no
user accounts, no authentication, and no persistent user data**. The trust
boundary is the hosting platform (a Hugging Face Space or a container you
control) — anyone who can reach the port can transcribe audio and read
`/health`.

In scope:

- Arbitrary code execution from a crafted audio file (decode path: libsndfile /
  ffmpeg / audioread)
- Path traversal or overwrite through uploaded filenames and session directories
- Resource exhaustion (memory, disk, CPU) from oversized or malicious uploads
- Leakage of `HF_TOKEN` or other environment secrets into logs or responses
- Supply-chain compromise of a pinned dependency or the optional YourMT3+
  checkpoint

Out of scope (by design, documented rather than fixed):

- Missing authentication or per-IP rate limiting — add a proxy if you expose
  this to the public internet beyond a demo
- Denial of service by simply posting many large-but-legal uploads (the queue is
  bounded, so this degrades to rejection, not memory exhaustion)
- Anything reached by an operator deliberately configuring unsafe values
  (`HOST`/`PORT`/token in `.env`)

## Built-in controls

| Control | Implementation |
|---|---|
| Upload limits | 50 MB and 5 minutes, extension allowlist, validated before decode |
| Request isolation | `mkdtemp` session directory per request, unique output ZIP |
| No info leak | users get a short message plus an `error_ref` correlation id; full detail stays in logs |
| Non-root, read-only container | `USER tabagent`, code mounted read-only, only cache/output dirs writable |
| Readiness gating | `/ready` returns 503 unless a required backend loads, so a broken deployment fails traffic |
| Secret handling | env-var only, never logged, `.env` excluded from the image via `.dockerignore` |
| Dependency pinning | `requirements.lock`, verified in CI and in the Docker build |
| Vulnerability gate | `tools/check_vulns.py`, fails on any advisory without a dated, justified acceptance in `.pip-audit-ignore` |
| Static analysis | `bandit` (SAST) + `gitleaks` (secret scan) in CI, both blocking |
| Model provenance | `YOURMT3_REVISION` pin + `YOURMT3_REQUIRE_PINNED_REVISION=1` fail-closed mode |

## Known accepted risks

See [`docs/ACCEPTED_RISKS.md`](docs/ACCEPTED_RISKS.md) for the current list, with
exposure analysis and expiry dates. Expired entries fail CI.

## Hardening the deployment

1. Run with `YOURMT3_REQUIRE_PINNED_REVISION=1` and a reviewed `YOURMT3_REVISION`.
2. Keep `HF_TOKEN` in the platform secret store, never in the image or `.env` in git.
3. Put the service behind a proxy that terminates TLS and applies rate limits if
   it is not a private/internal tool.
4. Re-run `tools/check_vulns.py` on a schedule — the acceptances expire
   (currently `2027-01-31`) and the gate is designed to force a re-review.
