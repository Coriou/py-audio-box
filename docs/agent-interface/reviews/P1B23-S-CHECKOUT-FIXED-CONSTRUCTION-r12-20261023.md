# P1B23 S checkout fixed construction r12 — actual handback

2026-10-23 · **STOP / construction NO-GO. ONE authorized host invocation consumed; raw host exit 1. No retry, fix, cleanup or acceptance. Fresh independent actual review is next.**

## Authority and verified inputs

Acted as the test sub-agent, without delegation. Fully read in bounded, untruncated chunks S-CHECKOUT-UMASK-SOURCE-r12-sol, S-CONSTRUCTION-DRIVER-SOURCE-r10-sol, S-HOST-LANE-SOURCE-r11-sol, S-SOURCE-RECIPE-SOURCE-r8-sol, last S-CHECKOUT-MODE-ACTUAL-r11-sol, original brief at RECOVERY-P1B2:2830, complete driver, recipe, construction script and Dockerfile. No applicable AGENTS.md was found in checked ancestry/scoped inventory. Memory registry search returned no relevant hit; no memory fact supports this report.

READONLY driver checkout remained clean at documentation ff70ae6fb9bdf7f2d978649ac60078554d4abb74; executable b968691a50bde1cc7d4165bb3fa36e8ade1612ce to HEAD changes only RECOVERY-P1B2. Application pin bd34b74cc0634c6e3b65cabb373ecbad3ccc2521 remains the frozen driver pin. Driver hash 1a496d513a763cf1b73ee0b74236439ed4b5a6fc5f549d3d4ca745ed25caee0d; Dockerfile hash d9de72ebc86c98ad5d056ce3da8c88eb1ab0cdfb1a3d2f32121eaee43c410ce2 verified before launch; driver hash and clean status rechecked afterward. Driver AST parsed without import.

Checked real nonsymlink canonical components of repository, driver, Dockerfile, review, private tmp and authorization; fresh evidence and authorization were initially absent. Authorization directory0700/leaf0600 and nonempty bounded review were checked. /tmp symlink text was private/tmp; physical parent identity matched /private/tmp (inode1000678450). Same global lane basename under both spellings was absent before launch and canonical lane remains absent afterward. No lane or alternate lock was acquired.

Host command python3 resolved to /usr/local/opt/python@3.14/bin/python3.14, real executable /usr/local/Cellar/python@3.14/3.14.7/Frameworks/Python.framework/Versions/3.14/bin/python3.14; CPython3.14.7, __debug__ true, optimize0. This is host startup identity only; required container Linux3.14.8/42 identities/decoder pins were UNREACHED.

## Exact single invocation

```text
python3 /Users/ben/.txcode/worktrees/py-audio-box/s-driver-checkout-mode-r12-20261023/docs/agent-interface/evidence/s-construction-driver.py --repository /Users/ben/Projects/py-audio-box --evidence /private/tmp/pab-s-checkout-fixed-construction-r12-20261023 --driver-sha256 1a496d513a763cf1b73ee0b74236439ed4b5a6fc5f549d3d4ca745ed25caee0d --dockerfile-sha256 d9de72ebc86c98ad5d056ce3da8c88eb1ab0cdfb1a3d2f32121eaee43c410ce2 --source-review /Users/ben/Projects/py-audio-box/docs/agent-interface/reviews/P1B23-S-CHECKOUT-UMASK-SOURCE-r12-sol.md --authorization /private/tmp/pab-s-checkout-fixed-authorization-r12-20261023/parent-authorization.txt
```

Direct unmodified driver invocation with shell-only private stdout/stderr/exit redirection; no reconstructed wrapper, candidate import preflight, recipe alternate or retry. Raw host exit is `1
`; stdout is exactly `STOP
`. Shell tool completion0 only reflects successful capture of the driver's exit and is not construction success.

## Actual failure and missing proof

At driver:183, first publish(invocation.json) enters driver:55 and raises `TypeError: Path.open() got an unexpected keyword argument 'opener'`. Path.open does not accept the opener keyword used by this reviewed driver. It fails before checkout dispatch, even before the scoped022 branch. The same publish implementation prevents STOP.json and final-state.json. evidence-hashes.json is consequently absent. Complete original traceback/state is retained in host.stderr, not reconstructed into substitute driver receipts.

Actual counts: ONE host invocation; ZERO external dispatches, checkout, recipe, Docker or inner construction. Inner raw exit UNREACHED, records0/52, primary completions0/9, actual/projection/restoration proof UNREACHED. No generated manifest, 263-copy/mode proof, resource/image admission, binary archive/LIVE equality, retention audit or full-ID cleanup receipt exists. Expected manifest66e702e5554fe1d198aa729385df67526441499207ade08f48bd3d2a8dfb8b46 is only a frozen requirement.

Stderr state: evidence acquired; checkout/context/recipe/lane/owners/base-tag/build/create/start/stop/remove all unattempted; container_id null; uncertain_commands empty; remaining_seconds null. Source control flow, lack of command receipts and exact residual entry set corroborate zero external dispatches. No Docker census or unknown-resource mutation was performed. No runtime resource/lifetime or cleanup obligation was acquired by this invocation; the three private retained files and authorization/capture sibling are residual proof. Existing historical checkouts/images remain untouched. Volatile /private/tmp is not indefinite retention.

## Deviations and retained inventory

The test agent's initial authorization writer also used Path.open(opener=...) and failed after creating only the empty0700 authorization directory. This was corrected solely in the agent-owned writer to built-in open with explicit0600; no driver invocation occurred in that preparation failure. The exact parent authorization was then retained, canonical paths/modes checked, and the sole driver invocation executed. The API issue was disclosed before launch; exact reviewed bytes were left unchanged. No cleanup or candidate correction followed.

Every retained leaf below was independently read and hashed after STOP; both parents0700 and all leaves regular0600. Retained driver/review/authorization match original inputs byte-for-byte. No generated driver inventory is claimed.

| Private retained path | Bytes | SHA256 |
|---|---:|---|
| `/private/tmp/pab-s-checkout-fixed-construction-r12-20261023/driver.py` | 29281 | `1a496d513a763cf1b73ee0b74236439ed4b5a6fc5f549d3d4ca745ed25caee0d` |
| `/private/tmp/pab-s-checkout-fixed-construction-r12-20261023/independent-source-review.txt` | 11800 | `53c65fe5c33a0b585bb5e044d04ff6b6b6980ca3e4d8adb5fec00cc952b7912f` |
| `/private/tmp/pab-s-checkout-fixed-construction-r12-20261023/parent-one-authorization.txt` | 2234 | `f4ef1bec9415de3ae9faf5f66919d7ee8ecbcc554b6e14f05f056f267ffed862` |
| `/private/tmp/pab-s-checkout-fixed-authorization-r12-20261023/host-argv.json` | 717 | `b8c332dcb8758f22828a5a4e3a266b8a0438b9b8d5f29927ef944d454b573ffd` |
| `/private/tmp/pab-s-checkout-fixed-authorization-r12-20261023/host.exit` | 2 | `4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865` |
| `/private/tmp/pab-s-checkout-fixed-authorization-r12-20261023/host.stderr` | 2973 | `c2136ba8adf7c423eb800a2087e102d5e179f394f03ecc66c13f096a823933af` |
| `/private/tmp/pab-s-checkout-fixed-authorization-r12-20261023/host.stdout` | 5 | `2af183be253fd571e13dd3d310357f3a1ccc7b18502a83ed918e0b88a1e5355f` |
| `/private/tmp/pab-s-checkout-fixed-authorization-r12-20261023/parent-authorization.txt` | 2234 | `f4ef1bec9415de3ae9faf5f66919d7ee8ecbcc554b6e14f05f056f267ffed862` |

## Remaining gates

No self-acceptance. Fresh independent actual-receipt review and parent decision remain required before any new source correction or separately authorized attempt. No commit, push, root integration, history mutation, DB/product maintenance, pytest or full255 occurred. Historical4 passes/1 failure/250 unrun and Sa;Ss; without Sf; RED remain unchanged. S/all255/native unwinding/FULL are not qualified. H qualification does not transfer; blocked-storage STOP is not waived. All other product/platform/release/activation gates remain open.
