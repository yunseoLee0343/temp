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
