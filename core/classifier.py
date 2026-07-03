# ============================================================
# ВЛАДЕЛЕЦ: P2 (классификатор сорта + обучение)
# TODO(P2):
#   - обучить на DATA_DIR (GPU/L4 доступен), сохранить веса в CLASSIFIER_WEIGHTS;
#   - довести macro-F1, приложить confusion matrix к отчёту;
#   - следить за дисбалансом (talc ~171 vs ~565/486): class weights + oversampling;
#   - сплит ТОЛЬКО по ID шлифа (GroupShuffleSplit) — иначе утечка и фальшивый F1.
# Это ЕДИНСТВЕННЫЙ способ отличить ordinary/hard_to_process (морфологии срастаний нет).
# ============================================================
"""
Ore-sort classifier: transfer learning (EfficientNet-B0 / ResNet50, ImageNet weights).

Non-negotiables baked in here (they cost real debugging to discover):
  * TRAIN/VAL SPLIT BY SLIDE ID via GroupShuffleSplit — never by individual image, or
    fields of one slide leak across the split and inflate F1.
  * Augmentations INCLUDE RandomGrayscale + ColorJitter — captures vary wildly in
    colour/lighting; without these the net learns colour, not mineralogy.
  * Class imbalance handled with inverse-frequency class weights (in the loss) AND a
    WeightedRandomSampler that oversamples the rare talc class.
  * Reports macro-F1 + confusion matrix at the end.

torch/torchvision are imported at module load, so import this module only where they're
installed (training + inference). analyze.py imports it lazily and degrades gracefully.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

import config as C
from core.labels import Sample, load_dataset_index

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Transforms  (grayscale + colorjitter are REQUIRED, not optional)
# --------------------------------------------------------------------------- #
def build_transforms(train: bool) -> transforms.Compose:
    norm = transforms.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD)
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(C.IMG_SIZE, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            # force colour-invariance: the model must key on texture/morphology, not hue
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            norm,
        ])
    return transforms.Compose([
        transforms.Resize(int(C.IMG_SIZE * 1.14)),
        transforms.CenterCrop(C.IMG_SIZE),
        transforms.ToTensor(),
        norm,
    ])


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class SlideDataset(Dataset):
    def __init__(self, samples: list[Sample], train: bool):
        self.samples = samples
        self.tf = build_transforms(train)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        img = Image.open(s.path).convert("RGB")
        return self.tf(img), C.CLASS_TO_IDX[s.label]


# --------------------------------------------------------------------------- #
# Split by slide ID  (the whole point)
# --------------------------------------------------------------------------- #
def group_split(samples: list[Sample], val_fraction: float = C.VAL_FRACTION,
                seed: int = C.RANDOM_SEED) -> tuple[list[Sample], list[Sample]]:
    groups = [s.slide_id for s in samples]
    gss = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(gss.split(samples, groups=groups))
    train = [samples[i] for i in train_idx]
    val = [samples[i] for i in val_idx]
    # sanity: no slide appears in both halves
    assert not ({s.slide_id for s in train} & {s.slide_id for s in val}), "slide leaked across split!"
    return train, val


# --------------------------------------------------------------------------- #
# Imbalance handling
# --------------------------------------------------------------------------- #
def class_weights(samples: list[Sample]) -> torch.Tensor:
    counts = Counter(s.label for s in samples)
    n = len(samples)
    w = [n / (len(C.CLASSES) * counts.get(c, 1)) for c in C.CLASSES]
    return torch.tensor(w, dtype=torch.float32)


def make_sampler(samples: list[Sample]) -> WeightedRandomSampler:
    counts = Counter(s.label for s in samples)
    per_sample = [1.0 / counts[s.label] for s in samples]  # oversample rare talc
    return WeightedRandomSampler(per_sample, num_samples=len(samples), replacement=True)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_model(backbone: str = C.CLASSIFIER_BACKBONE, num_classes: int = len(C.CLASSES)) -> nn.Module:
    if backbone == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif backbone == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    return m.to(DEVICE)


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #
def train(data_dir: Path = C.DATA_DIR, backbone: str = C.CLASSIFIER_BACKBONE,
          epochs: int = C.EPOCHS, batch_size: int = C.BATCH_SIZE, lr: float = C.LR,
          out_path: Path = C.CLASSIFIER_WEIGHTS) -> dict:
    samples = load_dataset_index(data_dir)
    if not samples:
        raise RuntimeError("No labelled images found — check DATA_DIR / folder names.")
    train_s, val_s = group_split(samples)
    print(f"[train] {len(train_s)} train imgs / {len(val_s)} val imgs "
          f"({len({s.slide_id for s in train_s})}/{len({s.slide_id for s in val_s})} slides)")

    train_loader = DataLoader(SlideDataset(train_s, train=True), batch_size=batch_size,
                              sampler=make_sampler(train_s), num_workers=C.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(SlideDataset(val_s, train=False), batch_size=batch_size,
                            shuffle=False, num_workers=C.NUM_WORKERS, pin_memory=True)

    model = build_model(backbone)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_s).to(DEVICE))
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    best_f1, best_state = -1.0, None
    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optim.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optim.step()
            running += loss.item() * x.size(0)
        sched.step()

        f1, _, _ = evaluate(model, val_loader)
        print(f"[train] epoch {ep:02d}/{epochs}  loss={running/len(train_s):.4f}  val_macroF1={f1:.4f}")
        if f1 > best_f1:
            best_f1, best_state = f1, {k: v.cpu() for k, v in model.state_dict().items()}

    # restore best + final report
    if best_state is not None:
        model.load_state_dict(best_state)
    f1, y_true, y_pred = evaluate(model, val_loader)
    print(f"\n[train] BEST val macro-F1 = {best_f1:.4f}")
    print("[train] confusion matrix (rows=true, cols=pred), order:", C.CLASSES)
    print(confusion_matrix(y_true, y_pred, labels=list(range(len(C.CLASSES)))))
    print(classification_report(y_true, y_pred, labels=list(range(len(C.CLASSES))),
                                target_names=C.CLASSES, digits=3, zero_division=0))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "backbone": backbone,
                "classes": C.CLASSES, "img_size": C.IMG_SIZE}, out_path)
    print(f"[train] saved weights -> {out_path}")
    return {"best_macro_f1": best_f1, "weights": str(out_path)}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> tuple[float, list[int], list[int]]:
    model.eval()
    y_true, y_pred = [], []
    for x, y in loader:
        logits = model(x.to(DEVICE))
        y_pred.extend(logits.argmax(1).cpu().tolist())
        y_true.extend(y.tolist())
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) if y_true else 0.0
    return f1, y_true, y_pred


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
_MODEL_CACHE: dict[str, tuple[nn.Module, list[str]]] = {}


def load_model(weights_path: Path = C.CLASSIFIER_WEIGHTS) -> Optional[nn.Module]:
    """Load trained weights, or return None if they don't exist yet (so analyze() can
    fall back to the rule-based verdict)."""
    weights_path = Path(weights_path)
    if not weights_path.exists():
        return None
    key = str(weights_path)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key][0]
    ckpt = torch.load(weights_path, map_location=DEVICE)
    model = build_model(ckpt.get("backbone", C.CLASSIFIER_BACKBONE))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    _MODEL_CACHE[key] = (model, ckpt.get("classes", C.CLASSES))
    return model


@torch.no_grad()
def predict(image: Union[str, Path, Image.Image, np.ndarray],
            model: Optional[nn.Module] = None) -> Optional[dict]:
    """Return {'verdict': cls, 'probs': {cls: p}} or None if no model is available."""
    if model is None:
        model = load_model()
    if model is None:
        return None
    if isinstance(image, (str, Path)):
        pil = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        # analyze() hands us BGR (OpenCV) arrays -> flip to RGB
        pil = Image.fromarray(image[..., ::-1] if image.ndim == 3 else image).convert("RGB")
    else:
        pil = image.convert("RGB")
    x = build_transforms(train=False)(pil).unsqueeze(0).to(DEVICE)
    probs = torch.softmax(model(x), dim=1).cpu().numpy()[0]
    idx = int(probs.argmax())
    return {"verdict": C.CLASSES[idx], "probs": {c: float(probs[i]) for i, c in enumerate(C.CLASSES)}}
