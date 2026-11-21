# Check what the model actually answers

Complete the existing 12 development prompts and the spider engineering control.
Use the same weights, tokenizer, and verbatim plain inputs.
Greedily generate at most 12 tokens without caching. This is bounded validation of the original prompts, not a new task family.
First-token predictions must match the completed pilot.

Save complete outputs and exact leading-target matches.
Check whether each purported hidden bridge appears in the generated text.
Do not count an intermediate name as evidence of hidden reasoning when the model simply answers with that name.
Do not silently replace a frozen source target that turns out to be wrong.

The source item `firstletter-populous-country` expects China and C.
The United Nations population estimates instead rank India first.
Flag that item separately. Preserve all original data and nominal scores.
Report any sensitivity calculation excluding the flagged item as post hoc.
