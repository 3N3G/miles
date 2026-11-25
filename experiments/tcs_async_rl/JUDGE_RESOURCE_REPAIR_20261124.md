# Validate resource-bound grading after the component check

The fresh forty-answer comparison did not qualify either Luna-high prompt.
The original reference prompt matched33/40 totals and96.21% of rubric items.
The component clarification matched36/40 totals and96.08% of items.
Both awarded one clear false full score. All132 reference, empty and irrelevant controls passed.
Item agreement averages the agreement fraction within each answer, then across answers.
These labels are assistant audits, not independent human judgments.

The false full score came from an oracle-runtime proof.
The candidate claimed that a bound proportional to N times a subpolynomial factor was O(N).
That implication is false when the factor grows.
Both graders supplied a sharper monomial count that the candidate had not established.
This is a substantive missing estimate in a rubric item requiring complete time accounting.
The original packet and pre-call labels remain frozen.
Some other disputes concern whether a direct alternative proves a reference-specific item,
or whether repeated misuse of the instance-set notation is a harmless slip.
They remain recorded as uncertain and do not erase the clear false-full error.

Reference prompt v4 adds one general paragraph to v3.
It checks asymptotic claims at the required scale and distinguishes a candidate's estimate from a grader-supplied improvement.
It also preserves a valid estimate when later unused discussion is wrong.
It gives an abstract subpolynomial-factor example, not a problem-specific answer or expected score.

Compare v3 and v4 on another disjoint40 saved answers:32 easy and8 medium across the same six questions.
Exclude all previous packets, audited proofs, and displayed seeded grades.
Freeze each Boolean label and reason before any new grading call.
Keep the original90% exact-score,95% case-balanced-item and no-clear-false-full gate.
Use exact-score agreement first, absolute error second. Do not retune against this fresh packet.
Run reference7, empty0 and irrelevant0 controls for every represented question and both prompts.
The existing shared ledger and comparison ceiling remain unchanged. No GPU job or solver generation is added.
If a prompt qualifies, regrade the existing384-answer screen with explicit new provenance before choosing the approved continuation.
