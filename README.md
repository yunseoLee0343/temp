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

`*.cubin` is ignored by Git. Push the textual artifacts and metadata:

```bash
git add results environment/version.txt
git commit -m "Add L4 Triton pipeline dumps"
git push origin main
```

Use the analogous H100 commit message after the H100 run.

Use the same Triton version for cross-GPU comparisons. The setup script records it in `environment/version.txt`.
