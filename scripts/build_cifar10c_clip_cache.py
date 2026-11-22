"""CIFAR-10 (+corruption shift, applied in code) via CLIP zero-shot -> pcpbench legacy 4-key
cache."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

from pcpbench.datasets import corruptions as _corruptions

CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)

_LEGACY_KEYS = ("source_logits", "source_labels", "target_logits", "target_labels")


def _prompt(class_name: str) -> str:
    return f"a photo of a {class_name.replace('_', ' ').lower()}"


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
          f"{output.with_name(output.stem + '_pcp.npz')} --name cifar10c --shift-family corruption")


def _apply_corruption(images, corruption: str, severity: int, seed: int):
    return _corruptions.apply_corruption_to_images(images, corruption, severity, seed=seed)


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


def run(root: Path, output: Path, *, corruption: str, severity: int, clip_model: str,
        pretrained: str, device: str, batch: int, seed: int, log=print) -> None:
    import open_clip
    import torch
    from datasets import load_dataset

    ds = load_dataset("uoft-cs/cifar10", cache_dir=str(root))
    classes = ds["train"].features["label"].names
    log(f"CIFAR-10 (HF): {len(classes)} classes | source=train(clean) "
        f"target=test(corrupted={corruption}, severity={severity}) | CLIP {clip_model}/{pretrained}")

    model, _, preprocess = open_clip.create_model_and_transforms(clip_model, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(clip_model)
    model.to(device)
    with torch.no_grad():
        tf = model.encode_text(tokenizer([_prompt(c) for c in classes]).to(device))
        tf = tf / tf.norm(dim=-1, keepdim=True)

    s_imgs = list(ds["train"]["img"])
    s_lbl = np.asarray(ds["train"]["label"], dtype=np.int64)
    t_imgs = _apply_corruption(list(ds["test"]["img"]), corruption, severity, seed)
    t_lbl = np.asarray(ds["test"]["label"], dtype=np.int64)
    log(f"  source {len(s_imgs)} imgs | target {len(t_imgs)} imgs")
    s_logits = _clip_logits(model, preprocess, tf, s_imgs, device, batch, log=log)
    t_logits = _clip_logits(model, preprocess, tf, t_imgs, device, batch, log=log)
    _save_cache(output, s_logits, s_lbl, t_logits, t_lbl)


def _selfcheck(out_path: Path | None = None) -> int:
    print("[SELF-CHECK] synthetic CLIP-shaped 10-class logits -> 4-key npz -> loads back.")
    rng = np.random.default_rng(0)
    K, per = 10, 40
    tmp = out_path if out_path is not None else (
        Path(tempfile.mkdtemp(prefix="cifar10c_")) / "cache.npz"
    )

    def logits(shift):
        L, y = [], []
        for c in range(K):
            base = np.full((per, K), -1.0)
            base[:, c] = 3.0 - (shift if c % 3 == 0 else 0.0)
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
    assert len(CIFAR10_CLASSES) == K
    print(f"[SELF-CHECK PASS] 4-key npz at {tmp} | K={K} matches len(CIFAR10_CLASSES)={len(CIFAR10_CLASSES)}.")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="CIFAR-10 (+corruption) CLIP zero-shot -> legacy 4-key cache.")
    p.add_argument("--root", type=Path, default=Path("data/cifar10"))
    p.add_argument("--output", type=Path)
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
    run(args.root, args.output, corruption=args.corruption, severity=args.severity,
        clip_model=args.clip_model, pretrained=args.pretrained, device=args.device,
        batch=args.batch, seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
