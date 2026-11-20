# Fixed-group completion within the MI210 job limit

Job 534098 finished its initial four responses at 28,515, 32,406, 34,253 and 39,958 tokens.
Their average is 33,783 generated tokens. All four received provisional zero grades in both orders.
The server's aggregate decoding rate is about 65 tokens per second.
If the remaining answers have that average length and rate, 32 responses need 4.62 hours of decoding.
This is a forecast from four responses, not a measured full-group duration.
The partition imposes a four-hour job limit. Model startup uses additional time.

## Exact resource repair

Use `group_check_v7_mi210_vllm8.json` in a fresh pinned checkout and run directory.
Increase only the maximum concurrent requests from four to eight.
Keep the same MI210 allocation, four-hour/four-GPU-hour cap, compatible image hash,
pinned weights, exact input/prompt, all 32 indexed seeds, sampling controls, thinking,
90,000 output and 98,304 context caps, eager target-model decoding, rubric and Luna medium.
Do not combine candidates from different runs into a nominal 32-response group.
Scheduling can change numerical samples. No bit-identical or performance-comparison claim is made.

The original server profiled 36.66 GiB of key/value cache, representing 1,177,061 tokens.
Eight trajectories at the fixed prompt and output cap need at most 722,992 token positions.
The replacement's recurrent-state allocation and cache profile must still be measured.
Require actual model load, cache admission and active decoding before cancelling the original own-session job.
Preserve every original completion, rubric receipt, config and failure/cancellation artifact.
The native MI350 SGLang job remains a separate runtime qualification path.

This repairs an observed inference-throughput budget mismatch under the user's standing repair authorization.
It does not change the question, reward or model to force nonconstant scores.
The RL gate still requires 32 completed, genuinely graded answers and at least two score values.
Human calibration and the independent-provider audit remain pending.
