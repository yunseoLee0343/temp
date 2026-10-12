# Standalone H100 descriptor/TMA control: 20261012T031235Z
This probe is **not an execution of vLLM upstream attention**.
It follows dynamic block-table -> tensor descriptor load -> reduction.
Check evidence.json for PTX `cp.async.bulk.tensor` and probe_result.json for correctness.
The pass dump is a pre-pass snapshot sequence, not automatically a LICM bug.
