# Input checks before the difficulty screen

I read all twelve selected problems, references and rubrics before new solver outputs existed.
The supplied reference arguments appear valid under their stated assumptions.
This is an assistant review, not independent human calibration.
The solver receives only the problem. Luna receives the problem, reference, rubric and extracted final answer.

Indices follow the [frozen screen order](evidence/difficulty_screen_selection_20261124.json), starting at one.

| Index | Problem | Reference mechanism checked |
|---|---|---|
| 1 | Auction surplus | Externality payments cancel the grand-bundle values, giving exactly two-thirds of optimal welfare. |
| 2 | Row-selection lower bound | Coordinate vectors plus the all-ones vector have one full-support, equal-magnitude dependence. Every deletion works. |
| 3 | Tree packing and cuts | Weighted averaging bounds crossings by cut size divided by packing weight. The fractional packing bound supplies the factor two. |
| 4 | Hypergraph removal counterexample | Epsilon0.99 tests only one-vertex templates. A dense bipartite host avoids the odd103-cycle and violates the upper bound. |
| 5 | Agreement hypergraph | Subtracting an anchor puts affine directions inside the restricted kernel. Empty edges contribute zero. |
| 6 | Communication selectors | Encode the entire selector vector in the exact ceiling bound, then simulate locally. Kappa1 costs zero bits. |
| 7 | Lipschitz shortest paths | A forced transition on two parallel edges has distance ratio growing linearly with grid refinement. |
| 8 | Cube face cycle | A reflected ternary ordering joins all nonempty faces. Its two vertex endpoints connect through the empty face. |
| 9 | Concentration inequality | Centered moment bounds and an elementary factorial estimate yield the exponential bound. Both tails and zero mean are handled. |
| 10 | Conditional functional inequalities | Average total variance or entropy over coordinates. Child energy bounds add exactly1/(k alpha_k) at each size. |
| 11 | Sorting estimator | One sampled order supplies the comparison oracle. Short transcript mass is polynomially small under the pointwise sampler bound. |
| 12 | Calibration and correlation | Conditional expectation controls the threshold classifier. Calibration and multiaccuracy each contribute at most2 tau. |

Two details matter for later rollout audits:

- The sorting problem also permits a simpler counting proof. At most2^(h+1) distinct binary strings have length at most h.
  This gives the required tail with a factor-two change in its unspecified constant, even without a prefix-free argument.
  A correct proof must not lose credit merely for using that valid route.
- The functional-inequality strategy rubric explicitly permits partial credit inside its two-point decomposition item.
  The approved binary-item protocol awards either zero or the full item weight.
  That coarsening is a limitation. Keep the selected reward fixed for this run.

No prompt, rubric, audit label, selected question or reward weight changed during these checks.
No new solver generation or paid call was needed.
