# Complete the censored generation controls

The frozen token-lens test's12-token completions begin with native thinking
text rather than answers. Treating truncation as wrong would invalidate the
capability gate. Preserve all initial readouts, scores and short prefixes.
This repair changes only the completion budget and native inference cache,
not questions, model, prompts, target labels or lens metrics. It is covered
by the user's inference-repair and autonomous five-GPU-hour authorization.

First run a diagnostic on the first two frozen arithmetic questions. Use the
same exact model_input and pinned model. Generate greedily with the model's
native cache, at most512 new tokens. Require the first12 token IDs to equal
the saved cache-disabled prefixes exactly. Also compute32 greedy tokens
without cache on the first question and require those to equal the cached
prefix. Print raw output including native content_thinking/content_text tags.
Abort on a mismatch. No extra activation lens reads occur here.

If that validation passes, complete the64 arithmetic/chain questions and
ten public practice questions from the same frozen input file. Keep the32
direct-answer control generations marked incomplete unless already ended.
For every repaired question require its saved uncached prefix to match the
cached output exactly. Save each completed item immediately. Max512 tokens,
greedy, no sampling seed. Questions touching the completion limit stay
censored. Do not change the original token-recovery endpoint or visit list.

For capability, parse the final content_text section and require the source
answer there. Use exact numeric value for arithmetic and case/spacing match
for practice text, plus explicit source-supported numeric words for7.
Do not count a target number merely mentioned during thinking as correct.
Record separately whether the intermediate occurs in any generated text,
including native thinking text. A hidden-intermediate claim requires a
completed correct output with no such occurrence. There is no LLM judge.

One diagnostic MI210 devel job, at most5 minutes (300 GPU seconds). Only if
its controls pass, one completion job at most30 minutes (1800 GPU seconds).
Reserve each before submission, one allocation at a time, within the
remaining five-hour program. Preserve failed runs and any partial results.
Archive raw outputs, input identities, controls, timings and logs.
