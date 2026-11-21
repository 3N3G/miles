# Reuse a late-pass decoder without advancing its semantic state

The first transport experiment predicts the final state from an earlier state.
That can hide an intermediate variable by mapping it to the final answer.
Test a different possibility before declaring a good predictor a good reasoning lens.

Train an output-coordinate map on pass three only, then reuse it at the same physical layer in earlier passes.
Shared model weights give a concrete reason for trying a common decoding basis.
Do not assume that they guarantee one.

Use the same frozen WikiText calibration rows, paragraph split, seeds, development prompts, and read budget.
No additional model forward passes are needed for this comparison.
Use the existing seed-zero fitted rotation and affine map at visit 35 for physical layer 11 in passes one and two.
Use the visit-41 map for its own final-core read. This preserves the fixed four sites: 11, 23, 35, and 41.

Compare two versions:

1. Apply the late-pass map unchanged.
2. Replace only its source mean and global scale with that of the receiving pass, using unlabeled calibration statistics.

The second version removes a prompt-independent phase offset without performing another model update.
It must still pass shuffled-pair and mean-only controls.
Evaluate final-state prediction and intermediate-concept recovery separately.
If it recovers a concept, test whether the same readout direction supports a meaningful causal swap.

This is a prespecified extension before inspecting the fitted transport readouts.
The remaining 43 source-bank test items stay reserved.
