# Repair the checkpoint screen startup validator

AMD screen546404 failed before generation after4 minutes32 seconds.
It produced zero solver answers, zero optimizer updates and zero new paid judge calls.
The error came from the new screen validator, not a weight mismatch.

The resumed startup deliberately publishes restored weights twice.
The second transfer checks that resetting and republishing the tensors preserves them.
Both transfers contained three engine checksum maps, each matching all1416 source tensor records.
The validator incorrectly required exactly one publication.

The repair verifies the exact event-history prefix restored from checkpoint108.
It then verifies both appended publications, their source iteration, and all three checksum maps.
It rejects missing/extra transfers, altered restored history and any changed checksum.
This avoids counting inherited publications as new transfers.

The actual failed-run event files passed the repaired check on the VM.
All142 TCS CPU tests, formatting checks and Ruff passed.
See`evidence/screen_publication_repair_20261124.json` for exact input fingerprints.

The replacement is`checkpoint_screen_high_v2.json`, with a new output directory.
Its maximum allocation is1 hour55 minutes on the same eight MI350 GPUs.
Combined with the272 seconds already used, this is at most15.9378 GPU-hours, below the approved16.
The twelve inputs, seed, model checkpoint, topology, sampling, prompt, judge and rubric remain fixed.
The old configuration, failed outputs and source checkpoint stay preserved.
The shared ledger remains572 settled calls and$0.5496883, with no pending reservations.

The existing approved inference-repair permission covers this replacement.
Record its actual job receipt immediately and supervise the remaining screen through completion.
