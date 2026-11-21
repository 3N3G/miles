# Numerical repair of the matched-direction control

The first matched-size stage failed its Jacobian-reference check, with maximum
logit discrepancy 0.1659336, exceeding the predeclared code tolerance 0.15.
The mathematical reflection was rewritten using a unit vector. Reordered FP32
operations changed BF16 rounding at the intervention, and some downstream logits.
The original plan required reproduction of the existing reference. Do not relax
that requirement or use the incomplete stage as a final comparison.

Repair: compute the Jacobian reference delta with the exact original expression
and operation order from `forward_with_swap`, including its final dose multiply.
Use that delta directly for the Jacobian condition. For other unit directions,
use its per-position norm and signed country coefficient. Require bitwise equality
of all reference logits, and retain the original 1e-5 pre-cast norm-match check.

The model, calibration fits, prompts, eligibility gate, methods, site, dose, seeds,
and random controls stay fixed. Preserve the failed directory and pinned source.
The replacement writes `matched-country-directions-v2`. It reruns only this stage,
within its existing ten-minute one-GPU budget. This is an inference/numerical repair
under the user's standing authorization. Archive both the failure and replacement.
