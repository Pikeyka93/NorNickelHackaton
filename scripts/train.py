# ============================================================
# ВЛАДЕЛЕЦ: P2 (обучение классификатора)
# TODO(P2): запустить на сервере (см. команду ниже), сохранить веса, приложить
#           macro-F1 + confusion matrix к отчёту/презентации.
# ============================================================
"""
Train the ore-sort classifier.

    python scripts/train.py --epochs 15 --backbone efficientnet_b0

Split is by SLIDE ID (GroupShuffleSplit) — see core/classifier.py. Prints macro-F1 and
a confusion matrix, saves best weights to CLASSIFIER_WEIGHTS (config.py).
"""
import argparse
import sys
from pathlib import Path

# make repo root importable regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as C  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Train ore-sort classifier")
    p.add_argument("--data-dir", default=str(C.DATA_DIR))
    p.add_argument("--backbone", default=C.CLASSIFIER_BACKBONE,
                   choices=["efficientnet_b0", "resnet50"])
    p.add_argument("--epochs", type=int, default=C.EPOCHS)
    p.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=C.LR)
    p.add_argument("--out", default=str(C.CLASSIFIER_WEIGHTS))
    args = p.parse_args()

    from core.classifier import train  # imported here so torch is only needed at train time
    train(data_dir=Path(args.data_dir), backbone=args.backbone, epochs=args.epochs,
          batch_size=args.batch_size, lr=args.lr, out_path=Path(args.out))


if __name__ == "__main__":
    main()
