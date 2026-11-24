# Two grading recipes still disagree with mathematical audits

Neither recipe reached the predeclared 90% exact-score and 95% rubric-item agreement gate.
The one prompt repair also failed on forty fresh answers. GPU screening and training were not submitted.

| Packet | Grading recipe | Exact score agreement | Item agreement | Mean score error, out of 7 |
| --- | --- | ---: | ---: | ---: |
| First 40 | Anonymous, two orders averaged | 32/40 (80%) | 96.0% | 0.275 |
| First 40 | Labeled reference, original prompt | 31/40 (77.5%) | 93.5% | 0.400 |
| Fresh 40 | Anonymous, two orders averaged | 29/40 (72.5%) | 95.8% | 0.275 |
| Fresh 40 | Labeled reference, repaired prompt | 32/40 (80%) | 94.5% | 0.400 |

Each packet contained 32 easy answers and eight answers from the original medium question.
They shared seventeen questions but no answer text or sample identities.
Both recipes used Luna medium, final answers only, and the original binary rubric weights.
The assistant froze all 176 item labels per packet before new outputs.
These are assistant agreement measurements, not human accuracy estimates.
Fifteen fresh answers had recorded ambiguity. Both recipes scored all eight fresh medium answers zero, matching the audit.
This still covers only one medium question.

One fresh proof correctly shows that the interval's lower endpoint is at least √(2n), exceeding its upper endpoint.
Both recipes withheld an item because it did not display the rubric's particular rearrangement, d≥n/δ.
Another candidate correctly computes a unary objective, but omits the density parameter choice.
The repaired judge gave zero, denying even that independent calculation because it used an arbitrary singleton subset.
These expose a remaining dispute between literal rubric steps and equivalent mathematical arguments.
They do not establish that every assistant label is right.
Two first-packet audit corrections remain separate from the frozen primary labels.

All reference, empty and irrelevant controls passed on all seventeen questions for every tested prompt.
The 444 new calls cost $0.4516942 at the recorded token rates, with no pending reservations.
Question-cluster uncertainty intervals are broad. No recipe passed the gate or established near-perfect agreement.
The proposed next decision is documented in [the options](JUDGE_NEXT_STEP_OPTIONS_20261124.md).
The twelve-hour RL allowance remains unused.

Details: [repair and original audit corrections](JUDGE_COMPARISON_REPAIR_20261124.md),
[fresh results](evidence/judge_comparison_v2_results_20261124.json),
[fresh intervals](evidence/judge_comparison_v2_intervals_20261124.json),
[final accounting](evidence/judge_comparison_final_accounting_20261124.json).

Private immutable artifacts: [first packet and receipts](https://huggingface.co/datasets/3N3G/internalization-runs/tree/c2b0d8f9c65b9162f7b316494132116a193ba4fd/miles/judge-comparison-v1-20261124),
[fresh packet, receipts, plot and ten-case review](https://huggingface.co/datasets/3N3G/internalization-runs/tree/05550135e6475e53c77ca409c4a9ea936a6e036e/miles/judge-comparison-v2-20261124).
All 603 and 633 artifact hashes, respectively, passed immutable readback checks.
