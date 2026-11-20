# Handle a recorded model token limit without hiding judge failures

The original native group contains a real model length limit: sample 25 generated 90,000 tokens without a final answer.
The training adapter would raise before calling the judge. This would crash an otherwise valid RL request.
The supplied rubric already gives an empty answer zero in both judge orders. Its empty-answer control has passed.

Keep the same final-answer rubric and actual Luna calls. Extend the training adapter to recognize Miles' `TRUNCATED` status.
If that status has no final answer, submit an empty candidate to the same rubric judge.
Require both returned empty-answer scores to be zero. A nonzero empty score fails the control.
Never assign a zero on a provider, transport or parser error. Those errors still stop the request.
An unfinished response without a recorded model token limit still raises an error.
Responses with nonempty final answers retain their existing scoring path.

Return a separate `generator_truncated` indicator for logging. The reward key remains the seven-point `reward` value.
Do not change any rubric item, weight, prompt, judge, effort, input, token budget or held-out split.
The raw model response remains available in the rollout artifact, including the original unfinished thinking.

The diagnostic gate remains unchanged: 32 nonempty final answers, passing controls and nonconstant real scores.
No old diagnostic grade or qualification result changes. No old response is merged into a replacement group.
This repair prevents a future training crash after that gate. It does not make the original group qualify.

Protocol version 2 exposes this capability in the local service health response.
The training driver and tunnel require that version before checkpoint conversion or training.
Validate a fresh VM service on a spare loopback port, using the preserved real truncated response.
Check a normal reference answer, the recorded truncation and an unclassified unfinished response.
The expected results are seven, a real empty-answer judgment of zero, and an HTTP error respectively.
Preserve the old service and receipts. After validation, replace only the service process owned by this session.
No new GPU job is needed. Provider credentials stay in the authorized VM client.

CPU controls verify both finish paths, per-request metadata, concurrent scoring, failed empty controls and provider-error propagation.
The live service repair passed its separate controls using the preserved failure text.
Reference scores were 7/7 in both orders. The recorded length-limit case scored 0/7 in both orders.
An unclassified unfinished response returned HTTP 422 without a judge call.
The new owned service now listens on the original loopback port and reports protocol version 2.
Its actual process and private receipt paths are in the operational records.
