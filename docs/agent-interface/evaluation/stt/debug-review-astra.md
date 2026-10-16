# Independent final debugging review — 2026-10-15

**GO for the bounded local repairs and their recorded mechanics results. NO-GO for general CPU settings, production/default promotion, or bilingual accuracy acceptance.** The evidence no longer supports rejecting this machine or either runtime merely because the original attempts failed.

Review was read-only apart from this file. It covered the original benchmark, diagnostic/supervision sources, raw stage logs, model/hash receipts, terminal container states, pull streams, fault capture/disassembly, portable compiler flags and fresh transcription JSON. No additional inference, pull, build or installation was run by this reviewer.

| Finding | Independent assessment |
| --- | --- |
| CT2 latency | Explicit OpenMP spread/cores placement completed the same small-model FR/EN pair twice with packed MKL retained; turbo also completed it. The baseline encoder **did** finish before generation stalled. A demonstrated local workaround is supported; the lowest-level scheduling interaction remains unidentified. |
| Pull supervision/provisioning | The inherited Docker alarm failure is reproduced. Final parent-supervised pull exited zero in 234.448 seconds, with the exact pinned linux/amd64 image verified. Earlier short windows caused repeated transfers; the original silent pull's precise historical cause remains unknown. |
| whisper.cpp binary | Captured SIGILL bytes and library offset match the saved EVEX instruction disassembly. Native compilation emitted an instruction unavailable on this guest. Explicit AVX2 compilation of the same embedded source completed the fixed pair. This is direct compatibility evidence, not a generic runtime rejection. |
| Successful cleanup | Successful CT2, portable-build and portable-pair receipts show client reaping, stopped/no-OOM states and removal. An independent final Docker inventory found no task containers in any state. Initial supervision/provisioning failures remain visible rather than being rewritten as successful runs. |

The portable tiny-model run produced fresh parsed output for 11.88 seconds of audio in 2.994 seconds of CLI aggregate time, including initialization; supervision took 3.435 seconds. Its French result has **10/11 normalized word edits**. Small CT2 has **2/11** on the same French utterance; turbo matches that single reference. These observations justify further accuracy evaluation, not a model/runtime ranking. Auto versus forced language, different models, timer scopes, host load, power constraints and concurrent downloading prevent a controlled retrospective speed comparison.

Two operational limits remain explicit. The inference supervisor's sequential Docker calls do not guarantee a hard total deadline when the daemon stalls. The initial baseline reap receipt is incomplete, although later process checks found no orphan. These limit reusable-harness certification, not the recorded successful outcomes.

Service continuity cannot be certified: the final inventory contains three running unrelated containers whereas the initial context recorded four. The final synthesis discloses this unexplained difference and scopes preservation statements to actions taken; it does not assert that all four stayed running.

The [synthesis](DEBUGGING.md), [CT2 report](debug-inference/REPORT.md), [pull report](debug-pull/REPORT.md), and [portable-build report](debug-cpp-build/REPORT.md) provide the detailed evidence. Retain the working profiles as local evaluation candidates; representative French, terminology, mixed speech, long/noisy media and other hardware remain unqualified.
