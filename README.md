# Triton state pipeline trace

This repository compiles two stateful Triton kernels through the NVIDIA pipeline and preserves:

`Python -> TTIR -> TTGIR -> LLVM IR -> PTX -> CUBIN -> SASS`

Targets are separated:

- **L4**: `sm89`
- **H100**: `sm90`

Both kernels are compiled with `num_stages = 1,2,3,4`.

## Lightning Studio

Lightning Studio already provides one default conda environment and may reject creation of a second venv. This repo therefore installs Triton into the Studio's current environment.

First setup:

```bash
git clone https://github.com/yunseoLee0343/temp.git
cd temp
bash scripts/setup_lightning.sh
```

On L4:

```bash
bash scripts/run_l4.sh
```

On H100:

```bash
bash scripts/run_h100.sh
```

Results are written to `results/L4-sm89/` or `results/H100-sm90/`.

Each kernel directory can contain `.ttir`, `.ttgir`, `.llir`, `.ptx`, `.cubin`, `.sass`, and `metadata.json`.

`*.cubin` is ignored by Git. Publish textual artifacts and metadata with:

```bash
bash scripts/commit_results.sh l4
```

or, after an H100 run:

```bash
bash scripts/commit_results.sh h100
```

The script stages only the selected target result directory plus `environment/version.txt` when present, creates a commit only when there are staged changes, and pushes to `origin/main`. It also fills the repository-local Git identity as `Yunseo Lee <lys139011@gmail.com>` if Lightning has not configured one.

Use the same Triton version for cross-GPU comparisons. The setup script records it in `environment/version.txt`.

## Explicit loop-pipelining experiment

A third kernel, `mini_chunk_fla_pipelined`, keeps the same loop-carried H-state recurrence but changes the chunk loop to:

```python
for c in tl.range(0, num_chunks, num_stages=PIPE_STAGES):
    ...
```

For each requested stage S, both the backend compile option `num_stages=S` and the loop-level `PIPE_STAGES=S` constexpr are used. The original `mini_chunk_fla` remains as the negative control where the backend option alone produced identical artifacts for S=1..4.

After a run, `tools/analyze_pipeline_retention.py` compares TTGIR SHA-256 values and reports stage-dependent `ttg.local_alloc`, `ttg.local_load`, async-related operations, barriers, and unique `!ttg.memdesc<...>` types. Reports are written under `results/<GPU>/analysis/`.


## Source-driven software-pipeliner probe

`mini_chunk_fla_forced` is designed directly from Triton's current pipeliner gates in:

- `AssignLatencies.cpp`
- `ScheduleLoops.cpp`
- `LowerLoops.cpp`
- `SoftwarePipeliner.cpp`

The probe differs from the previous kernel in one important way: the steady-state loop iterates only over `T // BT` full chunks, so the K/V/W loads have no runtime tail mask and no non-zero `other` value. This is intended to make a contiguous fp16 load width of at least 32 bits provable to `ModuleAxisInfoAnalysis`, satisfying `canBeConvertedToAsyncLoad()` and allowing non-zero load latency / stage distance to materialize.

The loop still carries H as a distance-1 recurrent state, while K and W have direct load-to-dot paths and V reaches the second dot through the value update.

Run only this source-driven experiment on H100 with:

```bash
git pull
bash scripts/run_h100_forced_pipeline.sh
```

The runner forces recompilation and enables Triton's official per-pass MLIR dump knobs:

```text
TRITON_ALWAYS_COMPILE=1
MLIR_ENABLE_DUMP=1
MLIR_DUMP_PATH=...
```

Outputs include final artifacts under:

```text
results/H100-sm90/stages-{1,2,3,4}/mini_chunk_fla_forced/
```

and raw pass-manager dumps under:

```text
results/H100-sm90/pass-dumps/
```

The pass-dump analyzer searches for latency/schedule attributes and the concrete lowering signatures we want to observe:

```text
ttg.async_copy_global_to_local
ttg.async_commit_group
ttg.async_wait
ttg.local_alloc
ttg.local_load
!ttg.memdesc<...>
```

It also records the `SoftwarePipeliner` internal LowerLoops / ExpandLoops dump markers when present. Publish the results with:

```bash
bash scripts/commit_results.sh h100
```
