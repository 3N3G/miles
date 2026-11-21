# Formatting diagnostic

The plain-completion capability check found incorrect or repetitive outputs on several source-bank prompts.
Test whether these failures persist with Inkling's official conversation format.
This is a diagnostic continuation of the inference validation. It does not replace or rescore the plain pilot.

Use the same model revision and unchanged user text for six named cases:
France, spider, photosynthesis, armistice, most-populous-country, and Great Wall.
Apply the checkpoint's official chat template with its default effort 0.9.
Append the native assistant text-channel prefix. Add no instruction to the user message.
Print and save the full render. Greedily generate at most 32 tokens, with no cache.
Use the same validated readout function, and save all results and failures.

The explicit assistant text prefix is a format condition, not a claim about the model's natural choice of channel.
This diagnostic cannot establish a general benchmark score or an improved lens.
