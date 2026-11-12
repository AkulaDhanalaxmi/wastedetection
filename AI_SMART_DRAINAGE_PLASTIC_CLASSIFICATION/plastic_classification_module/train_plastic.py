"""Train the lightweight plastic recyclability classifier (MobileNetV3 by default).
Trains on <processed>/train, selects the best epoch on <processed>/val (macro-F1), then fits temperature
and a confidence threshold on VAL only. The TEST split is never read here (use evaluate_plastic.py).
Usage:  python train_plastic.py --device cuda --out runs/plastic/default [--export-onnx]
NOTE: written for Colab/GPU or CPU; it has not been executed in the authoring sandbox (no torch there)."""
from __future__ import annotations

import argparse, hashlib, json, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from plastic_common import (IMAGENET_MEAN, IMAGENET_STD, IMG_EXT, confusion_matrix, expected_calibration_error,
                            fit_temperature, letterbox, load_yaml, open_rgb, per_class_prf, resolve, softmax,
                            suggest_threshold, sweep_thresholds)
from plastic_models import build_model


class Letterbox:
    def __init__(self, size): self.size = size
    def __call__(self, im): return letterbox(im, self.size)


def make_transforms(aug: dict, size: int, train: bool):
    norm = T.Normalize(IMAGENET_MEAN.tolist(), IMAGENET_STD.tolist())
    if not train:
        return T.Compose([Letterbox(size), T.ToTensor(), norm])
    return T.Compose([
        Letterbox(size), T.RandomHorizontalFlip(aug["hflip"]),
        T.RandomAffine(aug["rotate_deg"], translate=(aug["translate"],) * 2, scale=tuple(aug["scale"]), fill=114),
        T.ColorJitter(aug["brightness"], aug["contrast"], aug["saturation"], aug["hue"]),
        T.RandomApply([T.GaussianBlur(5, (0.1, 1.5))], p=aug["blur_p"]),
        T.ToTensor(), norm, T.RandomErasing(p=aug["erasing_p"], value=0)])


class CropFolder(Dataset):
    def __init__(self, root: Path, classes: list[str], tf):
        self.items = [(p, i) for i, c in enumerate(classes) for p in sorted((root / c).glob("*")) if p.suffix.lower() in IMG_EXT]
        self.tf = tf
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, y = self.items[i]
        return self.tf(open_rgb(p)), y


@torch.no_grad()
def predict(model, loader, device):
    model.eval(); outs, ys = [], []
    for x, y in loader:
        outs.append(model(x.to(device)).float().cpu().numpy()); ys.append(y.numpy())
    return np.concatenate(outs), np.concatenate(ys)


def macro_f1(logits, y, n):
    return float(np.mean([c["f1"] for c in per_class_prf(confusion_matrix(y, logits.argmax(1), n))]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-config", default="config/dataset_plastic.yaml"); ap.add_argument("--model-config", default="config/model_plastic.yaml")
    ap.add_argument("--data-dir", default=None); ap.add_argument("--out", default="runs/plastic/default")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=None); ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--export-onnx", action="store_true")
    a = ap.parse_args()
    dcfg, mcfg = load_yaml(a.dataset_config), load_yaml(a.model_config)
    tc, size, classes = mcfg["train"], mcfg["model"]["img_size"], dcfg["classes"]
    data = resolve(a.data_dir or dcfg["paths"]["processed_dir"]); out = resolve(a.out); out.mkdir(parents=True, exist_ok=True)
    random.seed(tc["seed"]); np.random.seed(tc["seed"]); torch.manual_seed(tc["seed"])

    tr = CropFolder(data / "train", classes, make_transforms(mcfg["augment"], size, True))
    va = CropFolder(data / "val", classes, make_transforms(mcfg["augment"], size, False))
    counts = np.bincount([y for _, y in tr.items], minlength=len(classes))
    if len(tr) == 0 or (counts == 0).any():
        raise SystemExit(f"Train split missing a class (counts {dict(zip(classes, counts.tolist()))}). Build the dataset first (docs/ANNOTATION_GUIDE.md).")
    if len(va) == 0:
        raise SystemExit("Validation split is empty; cannot select epochs or calibrate.")
    print("train counts", dict(zip(classes, counts.tolist())), "val", len(va))
    tl = DataLoader(tr, tc["batch_size"], shuffle=True, num_workers=tc["num_workers"], drop_last=len(tr) > tc["batch_size"])
    vl = DataLoader(va, tc["batch_size"], shuffle=False, num_workers=tc["num_workers"])

    dev = torch.device(a.device)
    model = build_model(mcfg["model"]["arch"], len(classes), mcfg["model"]["pretrained"] and not a.no_pretrained).to(dev)
    w = torch.tensor(counts.sum() / (len(classes) * counts), dtype=torch.float32).to(dev) if tc["class_weighting"] else None
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=tc["label_smoothing"])
    head = list(model.classifier.parameters()); head_ids = {id(p) for p in head}
    back = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW([{"params": head, "lr": tc["lr_head"]}, {"params": back, "lr": tc["lr_backbone"]}], weight_decay=tc["weight_decay"])
    epochs = a.epochs or tc["epochs"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    use_amp = bool(tc["amp"]) and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best, bad, hist = -1.0, 0, []
    for ep in range(epochs):
        for p in back: p.requires_grad = ep >= tc["freeze_backbone_epochs"]
        model.train(); tot, n = 0.0, 0
        for x, y in tl:
            x, y = x.to(dev), y.to(dev); opt.zero_grad(set_to_none=True)
            with torch.autocast(dev.type, enabled=use_amp):
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item() * len(y); n += len(y)
        sched.step()
        vlog, vy = predict(model, vl, dev); f1 = macro_f1(vlog, vy, len(classes)); acc = float((vlog.argmax(1) == vy).mean())
        hist.append({"epoch": ep + 1, "train_loss": tot / max(n, 1), "val_acc": acc, "val_macro_f1": f1}); print(hist[-1])
        if f1 > best:
            best, bad = f1, 0; torch.save({"state_dict": model.state_dict()}, out / "_best_tmp.pt")
        else:
            bad += 1
            if bad >= tc["patience"]: print("early stop"); break

    model.load_state_dict(torch.load(out / "_best_tmp.pt", map_location=dev)["state_dict"]); (out / "_best_tmp.pt").unlink()
    vlog, vy = predict(model, vl, dev)
    temp = fit_temperature(vlog, vy); probs = softmax(vlog, temp)
    sweep = sweep_thresholds(probs, vy)
    thr = suggest_threshold(sweep, mcfg["calibration"]["target_accepted_accuracy"], mcfg["calibration"]["min_coverage"])
    cm = confusion_matrix(vy, probs.argmax(1), len(classes))
    meta = {"arch": mcfg["model"]["arch"], "img_size": size, "class_names": classes, "temperature": temp,
            "suggested_threshold": thr, "threshold_note": "chosen on VAL; None = no threshold met the target, default_threshold is used",
            "val": {"n": int(len(vy)), "accuracy": float((probs.argmax(1) == vy).mean()), "macro_f1": macro_f1(vlog, vy, len(classes)),
                    "ece": expected_calibration_error(probs, vy), "confusion_matrix": cm.tolist(), "per_class": per_class_prf(cm), "threshold_sweep": sweep},
            "train_counts": dict(zip(classes, counts.tolist())), "seed": tc["seed"], "pretrained": mcfg["model"]["pretrained"] and not a.no_pretrained,
            "split_manifest_md5": hashlib.md5((data / "split_manifest.csv").read_bytes()).hexdigest() if (data / "split_manifest.csv").exists() else None,
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    torch.save({"state_dict": model.state_dict(), "meta": meta}, out / "best.pt")
    (out / "best.meta.json").write_text(json.dumps(meta, indent=2)); (out / "train_history.json").write_text(json.dumps(hist, indent=2))
    if a.export_onnx:
        model.eval().cpu()
        torch.onnx.export(model, torch.zeros(1, 3, size, size), str(out / "best.onnx"), input_names=["input"], output_names=["logits"],
                          dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
    print(f"saved {out/'best.pt'}  temperature={temp}  suggested_threshold={thr}\nVAL metrics only; run evaluate_plastic.py for held-out TEST metrics.")


if __name__ == "__main__":
    main()
