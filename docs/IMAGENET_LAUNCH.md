# Building the ImageNet-1k caches

The ten ImageNet-1k caches at K=1000 are scored zero-shot by OpenCLIP, like the CIFAR
caches. The source is the clean ImageNet-1k validation set; the target is the same images
corrupted at one severity. `scripts/run_imagenet_grid.sbatch` builds and converts all ten
on one node with 8 GPUs.

## Prerequisites

- `pip install open_clip_torch datasets` in the environment that runs the job, the same
  packages the CIFAR builders need.
- Access to <https://huggingface.co/datasets/ILSVRC/imagenet-1k>. The dataset is gated:
  request access on its page ahead of time, then log in with `huggingface-cli login` or set
  `HF_TOKEN`.
- About 6 GB for the validation split, downloaded once to `data/imagenet`.

## Smoke test

Build one cell first to measure peak memory and time at K=1000:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/build_imagenetc_clip_cache.py --root data/imagenet \
  --num-classes 1000 --corruption gaussian_noise --severity 3 --clip-model ViT-B-32 \
  --pretrained laion2b_s34b_b79k --device cuda --batch 256 \
  --output build/imagenet1k_vitB32_gaussian_noise_s3.npz
```

Each build holds the clean and the corrupted validation images in memory. The grid runs
`MAXJOBS` builds at a time (default 2); keep it at 1 if one build comes close to the
memory of the node.

## Launch

Submit the build and the leaderboard as two jobs, the second dependent on the first, so
the leaderboard never reads a partial `caches/` directory:

```bash
mkdir -p logs
JOBID=$(sbatch --parsable scripts/run_imagenet_grid.sbatch)
sbatch --dependency=afterok:"$JOBID" scripts/run_grid.sbatch
```

## Outputs

- `build/imagenet1k_<slug>_<corruption>_s<severity>.npz`: what the builder writes.
- `caches/imagenet1k_<slug>_<corruption>_s<severity>.npz`: the converted caches, ten in
  all. The build job checks each one (1,000 classes, 50,000 source and 50,000 target rows,
  finite logits) and exits with status 1 unless all ten pass.
- `outputs/<cache>_<method>.json` and `leaderboard.md`: written by `run_grid.sbatch`,
  which rebuilds the board from every row file under `outputs/`.
