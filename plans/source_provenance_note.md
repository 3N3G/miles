# Source provenance audit and submission repair

The early submission helper pinned and checked the repository HEAD at submission.
It ran jobs from a shared checkout. Later pulls could therefore change code before
a dependent job began. No completed plan or data/config was changed. This is a
source-provenance limitation, not a reason to infer a different scientific result.

Compatible additions while country jobs were queued added optional Jacobian rows,
country-pair and visit arguments, matched random doses, normalized raw/ridge
variants, and descriptive manifest fields. The existing default intervention
formula, doses, prompts, fitting artifacts, and validity criteria remained fixed.
The job registries preserve the exact submitted command. Stage-specific results
include native equalities and unchanged raw controls. No statistical inference
should rely on treating the submission HEAD as an immutable executed checkout.

From the next submission, the helper creates a detached worktree under the data
root at the exact submitted commit. Jobs run from that worktree. Later pulls into
the shared checkout cannot affect those jobs. This operational repair changes no
scientific protocol. Existing queued jobs are not cancelled or resubmitted merely
to improve provenance. Preserve their limitation in the detailed methods.
