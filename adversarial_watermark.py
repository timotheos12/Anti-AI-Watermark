#!/usr/bin/env python3
"""
adversarial_watermark_laion.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Same imperceptible adversarial watermark injector, but loads the LAION
watermark_model_v1.pt weights directly (no Hugging Face dependency).

FIX INCLUDED: the official LAION checkpoint was saved together with its
training optimizer's state (Google's "Scalable Shampoo" optimizer). Plain
torch.load() fails with `ModuleNotFoundError: No module named
'scalable_shampoo'` because pickle tries to reconstruct that optimizer
object too, even though we only need the model weights. This script uses
a tolerant unpickler that substitutes a harmless placeholder for any class
it can't import, then extracts just the real tensor weights afterward.

Install:
    pip install torch torchvision timm pillow numpy

Setup:
    Download watermark_model_v1.pt from:
    https://github.com/LAION-AI/LAION-5B-WatermarkDetection/releases/tag/1.0
    and place it next to this script (or pass --weights <path>).

Usage:
    python adversarial_watermark_laion.py photo.jpg protected.png
    python adversarial_watermark_laion.py photo.jpg protected.png --epsilon 6 --steps 300
"""

import argparse
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

try:
    import timm
except ImportError:
    sys.exit("[!] Missing dependency. Run:  pip install timm")

DEFAULT_WEIGHTS = "watermark_model_v1.pt"

WM_MEAN = [0.485, 0.456, 0.406]
WM_STD  = [0.229, 0.224, 0.225]
WM_SIZE = 224
WM_INDEX = 0   # class 0 = "watermark" in the LAION head


# ── Tolerant checkpoint loading ─────────────────────────────────────────────
# Fixes: ModuleNotFoundError: No module named 'scalable_shampoo'

class _DummyClass:
    """Stand-in for any class the unpickler can't import. We only need the
    plain tensors that make up the model weights, so anything pickle can't
    resolve (optimizer internals, etc.) is safely discarded as one of these."""
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)
    def __setstate__(self, state):
        pass
    def __reduce__(self):
        return (_DummyClass, ())


class _TolerantUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError, ImportError):
            return _DummyClass


class _TolerantPickleModule:
    """Drop-in replacement for the `pickle` module, passed to torch.load()
    via its `pickle_module=` argument."""
    Unpickler = _TolerantUnpickler
    Pickler = pickle.Pickler
    load = pickle.load
    dump = pickle.dump
    HIGHEST_PROTOCOL = pickle.HIGHEST_PROTOCOL
    UnpicklingError = pickle.UnpicklingError


def _extract_state_dict(obj):
    """Checkpoints vary in shape (bare state_dict, or wrapped in a dict with
    'model'/'state_dict'/etc. keys alongside optimizer state). Find the
    dict of {name: tensor} and drop anything that isn't a real tensor."""
    candidates = [obj]
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model", "net", "weights"):
            if key in obj:
                candidates.append(obj[key])

    for cand in candidates:
        if isinstance(cand, dict):
            tensors = {k: v for k, v in cand.items() if torch.is_tensor(v)}
            if len(tensors) > 5:
                return tensors
    sys.exit("[!] Could not find a usable weight dictionary inside the checkpoint.")


# ── Device ───────────────────────────────────────────────────────────────────

def pick_device(pref: str) -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Image helpers ────────────────────────────────────────────────────────────

def img_to_tensor(pil_img: Image.Image, device) -> torch.Tensor:
    arr = np.array(pil_img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def tensor_to_img(t: torch.Tensor) -> Image.Image:
    arr = t.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    return Image.fromarray((arr * 255.0).round().astype(np.uint8))


def normalize(x, device):
    m = torch.tensor(WM_MEAN, device=device).view(1, 3, 1, 1)
    s = torch.tensor(WM_STD,  device=device).view(1, 3, 1, 1)
    return (x - m) / s


# ── Perceptual masking ───────────────────────────────────────────────────────

def perceptual_mask(img: torch.Tensor) -> torch.Tensor:
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                      dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    gx = F.conv2d(lum, kx, padding=1)
    gy = F.conv2d(lum, ky, padding=1)
    grad = torch.sqrt(gx * gx + gy * gy + 1e-8)
    blur = torch.ones(1, 1, 5, 5, device=img.device, dtype=img.dtype) / 25.0
    grad = F.conv2d(grad, blur, padding=2)
    grad = grad / (grad.amax() + 1e-8)
    return 0.15 + 0.85 * grad


def total_variation(delta: torch.Tensor) -> torch.Tensor:
    dh = (delta[:, :, 1:, :] - delta[:, :, :-1, :]).abs().mean()
    dw = (delta[:, :, :, 1:] - delta[:, :, :, :-1]).abs().mean()
    return dh + dw


def yuv_weight_map(chroma_ratio: float, device) -> torch.Tensor:
    w_luma = torch.tensor([0.6, 1.0, 0.45], device=device).view(1, 3, 1, 1)
    return w_luma / chroma_ratio + (1 - 1 / chroma_ratio)


# ── Inference ────────────────────────────────────────────────────────────────

@torch.no_grad()
def watermark_prob(model, img_full, device):
    small = F.interpolate(img_full, size=(WM_SIZE, WM_SIZE), mode="area")
    nx = normalize(small, device)
    return F.softmax(model(nx), dim=-1)[0, WM_INDEX].item()


# ── Model ────────────────────────────────────────────────────────────────────

class WatermarkModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=2)

    def forward(self, x):
        return self.model(x)


def load_model(device, weights_path):
    if not os.path.isfile(weights_path):
        sys.exit(
            f"[!] Weights file not found: {weights_path}\n"
            f"    Download 'watermark_model_v1.pt' from:\n"
            f"    https://github.com/LAION-AI/LAION-5B-WatermarkDetection/releases/tag/1.0"
        )
    print(f"[*] Loading weights from {weights_path} …")

    raw = torch.load(
        weights_path, map_location="cpu",
        pickle_module=_TolerantPickleModule, weights_only=False,
    )
    state = _extract_state_dict(raw)
    print(f"    Found {len(state)} weight tensors in checkpoint.")

    model = WatermarkModel()
    cleaned = {}
    for k, v in state.items():
        nk = k.replace("module.", "")
        if not nk.startswith("model."):
            nk = "model." + nk
        cleaned[nk] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    matched = len(cleaned) - len(unexpected)
    print(f"    Matched {matched}/{len(cleaned)} tensors to the model "
          f"({len(missing)} missing, {len(unexpected)} unexpected).")
    if matched < len(cleaned) * 0.5:
        print("    ⚠  Less than half the weights matched — the architecture may not")
        print("       line up with this checkpoint. Confidence scores may be unreliable.")

    model = model.to(device).eval()
    return model


# ── Optimisation ─────────────────────────────────────────────────────────────

def optimise(model, orig, eps, steps, target, tv_weight, chroma_ratio, device):
    mask = perceptual_mask(orig)
    chan_w = yuv_weight_map(chroma_ratio, device)
    eps_map = eps * mask * chan_w

    delta = torch.zeros_like(orig).uniform_(-1e-3, 1e-3)
    delta = (orig + delta).clamp(0, 1) - orig
    best_delta, best_prob = delta.clone(), 0.0

    delta.requires_grad_(True)
    opt = torch.optim.Adam([delta], lr=eps / 12)

    for step in range(steps):
        opt.zero_grad()
        perturbed = (orig + delta).clamp(0, 1)
        small = F.interpolate(perturbed, size=(WM_SIZE, WM_SIZE), mode="area")
        logits = model(normalize(small, device))
        probs = F.softmax(logits, dim=-1)

        loss = -torch.log(probs[0, WM_INDEX] + 1e-8) + tv_weight * total_variation(delta)
        loss.backward()
        opt.step()

        with torch.no_grad():
            delta.clamp_(-eps_map, eps_map)
            delta.copy_((orig + delta).clamp(0, 1) - orig)

            prob = probs[0, WM_INDEX].item()
            if prob > best_prob:
                best_prob, best_delta = prob, delta.detach().clone()

            if step % 25 == 0 or step == steps - 1:
                bar = "█" * int(prob * 30) + "░" * (30 - int(prob * 30))
                print(f"    step {step+1:4d}/{steps}  [{bar}]  {prob:.4f}")

            if best_prob >= target:
                print(f"    target {target} reached at step {step+1}")
                break

    return best_delta, best_prob


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="LAION-weights adversarial watermark injection")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--epsilon", type=float, default=6.0)
    ap.add_argument("--steps",   type=int,   default=250)
    ap.add_argument("--target",  type=float, default=0.85)
    ap.add_argument("--tv",      type=float, default=0.08)
    ap.add_argument("--chroma",  type=float, default=2.0)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--device",  default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"[*] Device: {device}")
    eps = args.epsilon / 255.0

    print(f"[*] Loading {args.input}")
    try:
        orig_pil = Image.open(args.input).convert("RGB")
    except FileNotFoundError:
        sys.exit(f"[!] File not found: {args.input}")
    w, h = orig_pil.size
    print(f"    Size: {w}×{h}")

    model = load_model(device, args.weights)
    orig = img_to_tensor(orig_pil, device)

    base = watermark_prob(model, orig, device)
    print(f"\n[*] Baseline watermark confidence: {base:.4f}")
    if base < 0.05:
        print("    ⚠  Very low baseline on what may be a watermark-free test image is normal.")
        print("       But if this seems wrong on a known-watermarked test photo, the weight")
        print("       mapping above likely didn't line up — see the match ratio printed earlier.")

    print(f"\n[*] Optimising  ε≤{args.epsilon}/255 (masked)  "
          f"tv={args.tv}  chroma×{args.chroma}  {args.steps} steps\n")
    delta, final = optimise(model, orig, eps, args.steps, args.target,
                            args.tv, args.chroma, device)

    result = (orig + delta).clamp(0, 1)
    out_img = tensor_to_img(result)
    verified = watermark_prob(model, img_to_tensor(out_img, device), device)

    d = (delta.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0)
    mse = (d ** 2).mean()
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float("inf")

    print(f"\n[*] Confidence: {base:.4f} → {final:.4f} (verified {verified:.4f})")
    print(f"\n── Imperceptibility ──────────────────────")
    print(f"   Max Δ  : {np.abs(d).max():.2f}/255")
    print(f"   Mean Δ : {np.abs(d).mean():.3f}/255")
    print(f"   PSNR   : {psnr:.1f} dB  (>40 = imperceptible, >45 = excellent)")

    out_img.save(args.output)
    print(f"\n[✓] Saved → {args.output}")
    if verified < 0.5:
        print("[!] Below 0.5 — try --epsilon 8 --steps 400 --target 0.9")
    elif verified < args.target:
        print("[~] Below target but may still clear filters.")
    else:
        print("[✓] Clears typical pwatermark thresholds (0.3–0.5) with margin.")


if __name__ == "__main__":
    main()
