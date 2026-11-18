# Why the vulnerability gate goes red, and what else is still open

Verified against `main` at `110e045` ("Address transcription audit findings and restore
validation gates", #14) on 2026-11-18. Every claim below has the command that produced it.

---

## 1. TL;DR

The gate does not have a vulnerable dependency. It has an **identity-matching bug**:
`tools/check_vulns.py` accepts an advisory only when the *single* `id` pip-audit reports
appears in `.pip-audit-ignore`, but an advisory has several identifiers
(`PYSEC-…`, `GHSA-…`, `CVE-…`) and pip-audit collapses them by alias set, keeping whichever
record it saw **first**. The ledger is written in `PYSEC-…` spellings; when OSV hands back
the `GHSA-…` record first, ten already-accepted torch advisories look unaccepted and the
build fails without anything having changed.

That is why the same tree was green at 01:55 and red at 02:04 with no code, lockfile or
ledger change in between, and why three runs from another branch were red while `main` was
green.

**Fix:** match on any identifier the advisory carries (id ∪ aliases), not on `id` alone.
Branch `arena/91504cfe-tab-agent-pro` already implements exactly that in `053abd1`
("Match advisories by any identifier so CI stops failing on renamed IDs", PR #15, open).

---

## 2. Evidence

### 2.1 The failing step

`gh run view 44220369043 --json jobs` →

```
Dependency vulnerability scan (shipped set)   failure   (every earlier step: success)
Type check with mypy / pytest / simulations   skipped
```

So lint, black, index, env, docker, lockfile --static, secret scan and bandit all passed;
the failure is inside `python tools/check_vulns.py`, which exits 1 when any reported
advisory is not covered by an unexpired `.pip-audit-ignore` entry.

### 2.2 pip-audit collapses aliases and keeps the first id

`pip-audit 2.10.1`, `pip_audit/_audit.py:55-93`:

> "Individual vulnerability results are uniqued based on their `aliases` sets: any two
> results for the same dependency that share an alias are collapsed into a single result
> with a union of all aliases."

and `pip_audit/_service/interface.py:139-146`:

```python
def merge_aliases(self, other):
    aliases = self.aliases | other.aliases - {self.id}
    return replace(self, aliases=aliases)   # `self.id` survives
```

`replace(self, …)` keeps **`self`'s id**, i.e. the first record returned. Whether the
surviving id is `GHSA-…` or `PYSEC-…` is decided by advisory-server ordering, not by this
repository.

### 2.3 The ledger is keyed on one spelling only

`grep -vE '^\s*(#|$)' .pip-audit-ignore | awk '{print $1}'` → 25 entries, of which the
torch ones are all `PYSEC-2025-41/189/190/191/192/193/194/195/198/203/204/205/206/207/208/209`,
`PYSEC-2026-139/1856/1970/2286`. The only `GHSA-…` entry in the file is setuptools'
`GHSA-xmj3-9gf7-h52w`.

### 2.4 Those ten GHSA ids are the same ten advisories, under other names

`gh api /advisories -f ecosystem=pip -f affects=torch@2.5.1` →

| GHSA | severity | CVE | accepted in the ledger as |
|---|---|---|---|
| GHSA-53q9-r3pm-6pq6 | **critical** | CVE-2025-32434 | PYSEC-2025-41 |
| GHSA-63cw-57p8-fm3p | high | CVE-2026-24747 | PYSEC-2026-2286 |
| GHSA-887c-mr87-cxwp | medium | CVE-2025-3730 | PYSEC-2026-1970 |
| GHSA-vgrw-7cvw-pwgx | medium | CVE-2025-2999 | PYSEC-2025-193 |
| GHSA-f4hp-rmr7-r7v8 | medium | CVE-2025-2998 | PYSEC-2025-192 |
| GHSA-3749-ghw9-m3mg | low | CVE-2025-2953 | PYSEC-2025-191 |
| GHSA-x3gm-94wq-g975 | low | CVE-2025-2149 | PYSEC-2025-190 |
| GHSA-c678-jfcj-6jmf | low | CVE-2025-2148 | PYSEC-2025-189 |
| GHSA-rrmf-rvhw-rf47 | low | CVE-2025-3000 | PYSEC-2025-194 |
| GHSA-qfhq-4f3w-5fph | low | CVE-2025-3001 | PYSEC-2025-195 |

Ten GHSA spellings, ten matching `PYSEC` entries in the ledger, zero GHSA spellings in it.
`setuptools==80.10.2` has the same problem with `GHSA-h35f-9h28-mq5c` (= `PYSEC-2026-3447`,
which *is* in the ledger), so the reported count can be 10 or 11 depending on ordering.

### 2.5 Timeline: identical inputs, opposite verdicts

| run | branch | when | vuln gate |
|---|---|---|---|
| 44219495789 | `arena/18940913` (`cccac32`) | 01:55Z | **success** |
| 44220369043 | `arena/18940913` (`d30aae1`, docs-only delta) | 02:04Z | **failure** |
| 44221622923 | `main` (`110e045`) | 02:52Z | success |
| 44221548243 / 44223762871 / 44224901017 | `arena/91504cfe` | 14h/13h ago | **failure** ×3 |

`d30aae1` differs from `cccac32` only in `RELIABILITY_REVIEW.md` and `audit_poc/`. The
lockfile and ledger were byte-identical across the first two rows, and no dependency
changed between 02:04 and the green `main` run — so the difference cannot be a new
advisory. It is which record won the alias merge.

An independent group hit the same wall: `053abd1` on `arena/91504cfe` is titled *"Match
advisories by any identifier so CI stops failing on renamed IDs"* and its docstring names
`PYSEC-2025-191` → `GHSA-3749-ghw9-m3mg` as the case in point.

---

## 3. What is *not* the cause (checked and ruled out)

| Hypothesis | Method | Verdict |
|---|---|---|
| An expired accepted-risk entry | all 25 entries expire `2027-01-31`; the script prints `EXPIRED` distinctly | not it |
| A genuinely new advisory | full PyPI advisory sweep over all 118 lock pins: only `torch` and `setuptools` carry advisories, and both sets are already in the ledger | not it |
| The docs-only diff | the failing step is untouched by the diff; `main` passes the same step | not it |
| A timeout/flake in the OSV call | a *transport* failure prints `❌ vulnerability scan failed to run` and returns 1, which is indistinguishable from a finding in CI (see §4.2) | possible, but the alias mismatch explains the persistent reds on `91504cfe` |

---

## 4. Defects in the gate itself, beyond the immediate red

1. **Identity matching (the red).** Match `id` ∪ `aliases`; or key the ledger by CVE. Code
   fix preferred — adding ten `GHSA-…` lines re-breaks the next time the feed renames
   something.
2. **"Scan failed to run" is fatal, and indistinguishable from "found something".** A 429,
   a DNS hiccup or an OSV outage fails CI with the same exit code as a real finding. A
   retry with backoff, or a distinct exit status/annotation, would stop this class of
   false red.
3. **The docstring promises severity filtering that does not exist.** `check_vulns.py`
   says it "fails on HIGH/CRITICAL findings"; the loop fails on *every* reported advisory,
   severity-blind. Either pass severity through from the advisory payload or fix the
   comment — right now a low-severity cosmetic advisory blocks a release while the
   documentation claims otherwise.
4. **The ledger is hand-maintained and drifts.** #14 removed the duplicate
   `PYSEC-2026-1970`, `PYSEC-2025-191`, `PYSEC-2025-41` lines that the old file carried —
   the same duplicates are the fingerprint of this bug: someone kept adding spellings by
   hand. Nothing tests the ledger's coverage; `tests/test_check_vulns.py` on `91504cfe` is
   the first test this tool has had.

---

## 5. Other live issues, re-verified today

Status against `110e045`. "Harness" is `audit_poc/`.

| # | Finding | Status now | Evidence |
|---|---|---|---|
| T-1 | simultaneous chord forced onto one string | **fixed** | `repro.py`: pitch 64+67 → `[(3,9,'pick'),(4,8,'pick')]`; only one strict `xfail` remains in `tests/`, and it is the octave case |
| T-2 | techniques fabricated from timing alone | **mostly fixed** | sweep at sensitivity 0.7: gaps 0.125/0.150/0.300 and simultaneous → `pick`; the window is now `0.05+0.7·0.25` **and** `0 ≤ gap ≤ 0.03`. Residual: a 70 ms gap with 50 ms notes is still labelled `slide` (geometry-only, no bend data exists) |
| T-3 | dropped notes invisible in exports | **fixed** — my repro was wrong | `agent.last_skipped_events` → `[{'pitch': 30, …, 'reason': 'out_of_range'}]` appears in JSON `skipped_notes`, `.tab` gets `Warning: 1 detected note(s) omitted from the tab.`, CLI prints to stderr. Old repro called `export_tab_to_json()` without the argument the CLI passes; corrected in this branch |
| R-1 | blocking `read("*l")` inside `reaper.defer` | **still broken** | probe B: one `poll()` blocked the main thread **3.00 s** on a quiet child; `reaper/TabAgent.lua:220` |
| R-2/R-3/R-4/R-6/R-7 | undo pairing, wrong project, shell interpolation, default profile, output dir | **fixed** | probe A shows only `MB: Tab Agent — Project Changed` after a switch; probe C's sentinel is not created; `tests/test_reaper_contract.py` covers the rest |
| R-5 | CSV settings dialog shifts fields | **fixed** | `reaper/Settings.lua:81` preserves empty fields and rejects `<9` parts. Residual: a comma *inside* a value still shifts (inherent to `GetUserInputs`) |
| R-8 | no cancel, no lock during a run | **still open** | dialogs are error-only; nothing prevents a second run or a mid-run edit |
| W-1 | upload size enforced only after the bytes land | **still open** | `gr.Blocks(...)` at `app.py:591` passes no `max_file_size`, so the upload route's limit stays `math.inf`; `MAX_FILE_SIZE_MB` is checked in `_validate_audio` *after* Gradio has written the temp file. A direct `POST /gradio_api/upload` never reaches the check |
| M-2 | zero-note run exits 0 | **partly fixed** | probe B: `exit code: 0`, `files produced: []`, but now prints `⚠️ No notes were transcribed`. Automation still sees success |
| M-4 | Suno octave cleanup is O(n²) | **still quadratic** | 1k→0.075 s, 2k→0.315 s, 4k→1.372 s, 8k→6.06 s (4× per doubling) |
| D-x | ReaPack ships a script five commits stale | **newly confirmed** | `index.xml` pins `commit="4061696b"` at version **1.0.1** while `pyproject.toml` is **1.1.0**; `4061696b` is #8, i.e. *before* the streaming rewrite — no `reaper.defer` at all (REAPER freezes for the whole run), no project guard, no report-line protocol. `tools/check_index.py` only checks that the sha exists, so CI stays green |
| V-1 | gates that cannot fail | **improved** | the chord/tech/demucs guards are now real assertions; `test_transcription_safety.py:129` is the last strict `xfail` |

### Harness rot fixed in passing

* `audit_poc/reaper_probe.py` returned `GetMediaSourceFileName` as its *second* value; the
  Lua binding returns it as the only value, so every REAPER probe aborted at "Glue
  Required" (fixed).
* `audit_poc/pipeline_probe.py`'s `FakeSplitter` lacked `separation_mode`, which `main.py`
  reads, so the whole run died with `AttributeError` (fixed).
* `audit_poc/repro.py` case 3 called `export_tab_to_json()` without `skipped_notes`, so it
  "reproduced" a defect that no longer exists (fixed).

---

## 6. Suggested order

1. Land the alias-aware matching (`053abd1`, PR #15) — it is the only thing standing
   between `main` and a repeat of the red, and it needs the test it brings.
2. Make "scan failed to run" retry, and make severity mean what §4.3 says it means.
3. Refresh `index.xml` to the current commit and version; have `check_index.py` fail when
   the pin is behind `HEAD` for a shipped script. Right now the ReaPack channel delivers
   the pre-#11 REAPER script, which is the blocking one.
4. R-1: read the child's output on a timer/`io.read` slice instead of blocking inside
   `defer`; it is the last finding that stalls the user's REAPER session.
5. W-1: set `max_file_size` on `gr.Blocks` (and keep `_validate_audio` as the second line
   of defence).
