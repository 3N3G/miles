# Eight additional saved proofs under the clarified grader

This diagnostic checks whether the frozen revised grader awards an incorrect proof or rejects a sound proof over harmless wording.
It uses eight saved Qwen3.5-4B answers from eight questions in the existing twenty-question easy training bundle.
No solver generation or GPU training ran for this check.

Selection used seed 20261125 and excluded the sixteen earlier explicit prompt-adjudication cases.
Assistant decisions for all 35 rubric items were recorded and fingerprinted before new judgments.
The existing Luna-medium service graded both anonymous orders with unchanged prompt v3 and rubric weights.

All **70/70 repeated item decisions** match those assistant labels.
Seven answers adjudicated valid receive 7/7. The false theorem assertion receives 0/0.
All sixteen paired-reference grades are seven. These are eight answers, not seventy independent examples.

| Same saved answer | Original score pair | Revised score pair |
| --- | --- | --- |
| Universal hashing proof | 7, 6 | 7, 7 |
| Communication protocol proof | 7, 5 | 7, 7 |
| False theorem assertion | 0, 0 | 0, 0 |

For the communication proof, jointly encoding the index vector meets the sharp bit bound.
When its size parameter equals one, the vector is unique and needs zero communication.
The other five score pairs remain 7/7. The model answers were fixed throughout.

The sixteen calls cost an estimated **$0.0177312**.
Combined qualification cost is now **$0.4150215** under the existing shared $5 cap, with zero pending reservations.
Assistant labels are not human calibration. The cases use training questions, and stochastic judge variation remains a confound.
Earlier borderline disagreements remain preserved. This check does not measure model improvement.

The [verified private evidence](https://huggingface.co/datasets/3N3G/internalization-runs/tree/5e04681b64bccd11e840eac5e8e8d2545e1f6eb4/miles/easy-overfit-v14-additional-proof-audit-20261122)
contains all eight proofs, pre-call labels, judgments, exact requests, and comparisons.
See the [audit plan](JUDGE_ADDITIONAL_AUDIT_PLAN_20261122.md) and [review client](review_saved_judge.py).
