"""ImageNet-C (fixed class/corruption/severity subset) via CLIP zero-shot -> pcpbench legacy
cache."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

from pcpbench.datasets import corruptions as _corruptions

_LEGACY_KEYS = ("source_logits", "source_labels", "target_logits", "target_labels")


def _prompt(class_name: str) -> str:
    return f"a photo of a {class_name.replace('_', ' ').lower()}"


def _apply_corruption(images, corruption: str, severity: int, seed: int):
    return _corruptions.apply_corruption_to_images(images, corruption, severity, seed=seed)


def _save_cache(output: Path, s_logits, s_lbl, t_logits, t_lbl) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(output),
             source_logits=s_logits.astype(np.float32), source_labels=s_lbl.astype(np.int64),
             target_logits=t_logits.astype(np.float32), target_labels=t_lbl.astype(np.int64))
    print(f"  saved {output} | source (clean) {s_logits.shape} | target (corrupted) {t_logits.shape} "
          f"| K={s_logits.shape[1]}")
    print(f"  CLIP zero-shot top-1: source {(s_logits.argmax(1) == s_lbl).mean():.3f} | "
          f"target {(t_logits.argmax(1) == t_lbl).mean():.3f}")
    print(f"  next: python -m pcpbench.datasets.convert --in {output} --out "
          f"{output.with_name(output.stem + '_pcp.npz')} --name imagenetc --shift-family corruption")


def _clip_logits(model, preprocess, text_feat, images, device, batch, log=print):
    import torch

    model.eval()
    scale = model.logit_scale.exp().item()
    out = []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            tens = [preprocess(im) for im in images[i:i + batch]]
            f = model.encode_image(torch.stack(tens).to(device))
            f = f / f.norm(dim=-1, keepdim=True)
            out.append((scale * f @ text_feat.T).cpu().numpy())
            if i % (batch * 20) == 0:
                log(f"    {i}/{len(images)} images")
    return np.concatenate(out).astype(np.float64)


def run(root: Path, output: Path, *, num_classes: int, corruption: str, severity: int,
        clip_model: str, pretrained: str, device: str, batch: int, seed: int, log=print) -> None:
    """Build the ImageNet-C-subset CLIP cache from the REAL, gated ILSVRC/imagenet-1k dataset."""
    import open_clip
    import torch
    from datasets import load_dataset

    log(f"  CLIP {clip_model}/{pretrained} | device={device}")
    log(f"WARNING: downloading/caching the ImageNet-1k validation split (~6 GB) to {root} on "
        f"first run -- requires HuggingFace auth (huggingface-cli login or HF_TOKEN) AND access "
        f"granted at https://huggingface.co/datasets/ILSVRC/imagenet-1k (see docs/IMAGENET_LAUNCH.md).")

    val = load_dataset("ILSVRC/imagenet-1k", split="validation", cache_dir=str(root))

    label_names = getattr(val.features["label"], "names", None)
    if not label_names:
        raise RuntimeError(
            "ILSVRC/imagenet-1k validation split's `label` feature has no `.names` list of real "
            "class text -- refusing to fall back to placeholder 'class_i' prompts, which would "
            "make CLIP zero-shot logits meaningless. Inspect `val.features['label']` and fix the "
            "class-name lookup before proceeding."
        )
    class_names = list(label_names)
    if len(class_names) != 1000:
        raise RuntimeError(
            f"expected 1000 ImageNet classes from val.features['label'].names, got "
            f"{len(class_names)} -- refusing to proceed with a mismatched class list."
        )

    val = val.filter(lambda ex: ex["label"] < num_classes)
    kept = len(val)
    log(f"SUBSET: using {num_classes} of 1000 ImageNet classes (labels 0..{num_classes - 1}), "
        f"{kept} of 50000 val images; dropped the other {1000 - num_classes} classes")

    classes = class_names[:num_classes]
    model, _, preprocess = open_clip.create_model_and_transforms(clip_model, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(clip_model)
    model.to(device)
    with torch.no_grad():
        tf = model.encode_text(tokenizer([_prompt(c) for c in classes]).to(device))
        tf = tf / tf.norm(dim=-1, keepdim=True)

    s_imgs = list(val["image"])
    s_lbl = np.asarray(val["label"], dtype=np.int64)
    log(f"  source (clean) {len(s_imgs)} imgs")

    t_imgs = _apply_corruption(s_imgs, corruption, severity, seed)
    t_lbl = s_lbl.copy()
    log(f"  target (corrupted={corruption}, severity={severity}) {len(t_imgs)} imgs")

    s_logits = _clip_logits(model, preprocess, tf, s_imgs, device, batch, log=log)
    t_logits = _clip_logits(model, preprocess, tf, t_imgs, device, batch, log=log)
    _save_cache(output, s_logits, s_lbl, t_logits, t_lbl)


def _selfcheck(out_path: Path | None = None) -> int:
    print("[SELF-CHECK] synthetic CLIP-shaped 20-class logits -> 4-key npz -> loads back.")
    K, per = 20, 30
    num_classes, corruption, severity = K, "gaussian_noise", 3
    print(f"SUBSET: using {num_classes} of 1000 classes, corruption={corruption}, "
          f"severity={severity} (dropped the rest)")

    rng = np.random.default_rng(0)
    tmp = out_path if out_path is not None else (
        Path(tempfile.mkdtemp(prefix="imagenetc_")) / "cache.npz"
    )

    def logits(shift):
        L, y = [], []
        for c in range(K):
            base = np.full((per, K), -1.0)
            base[:, c] = 3.0 - (shift if c % 4 == 0 else 0.0)
            L.append(base + rng.standard_normal((per, K)) * 1.4)
            y += [c] * per
        return np.concatenate(L), np.array(y, np.int64)

    sL, sy = logits(0.0)
    tL, ty = logits(3.0)
    _save_cache(tmp, sL, sy, tL, ty)

    z = np.load(tmp, allow_pickle=False)
    for key in _LEGACY_KEYS:
        assert key in z.files, f"selfcheck cache missing '{key}'"
    assert z["source_logits"].shape[1] == K, z["source_logits"].shape
    assert z["target_logits"].shape[1] == K, z["target_logits"].shape
    print(f"[SELF-CHECK PASS] 4-key npz at {tmp} | K={K}.")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="ImageNet-C (fixed class/corruption/severity subset) CLIP zero-shot -> legacy 4-key cache.")
    p.add_argument("--root", type=Path, default=Path("data/imagenet"))
    p.add_argument("--output", type=Path)
    p.add_argument("--num-classes", type=int, default=100,
                    help="Fixed subset size: first N of the 1000 ImageNet classes.")
    p.add_argument("--corruption", type=str, default="gaussian_noise",
                    choices=list(_corruptions.CORRUPTIONS))
    p.add_argument("--severity", type=int, default=3, choices=[1, 2, 3, 4, 5])
    p.add_argument("--clip-model", type=str, default="ViT-B-32")
    p.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--selfcheck", action="store_true")
    p.add_argument("--selfcheck-out", type=Path, default=None,
                    help="Optional path for the selfcheck to write its synthetic 4-key npz to "
                         "(defaults to a tempdir).")
    args = p.parse_args(argv)
    if args.selfcheck:
        return _selfcheck(args.selfcheck_out)
    if args.output is None:
        print("ERROR: pass --output (or --selfcheck).", file=sys.stderr)
        return 2
    run(args.root, args.output, num_classes=args.num_classes, corruption=args.corruption,
        severity=args.severity, clip_model=args.clip_model, pretrained=args.pretrained,
        device=args.device, batch=args.batch, seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
