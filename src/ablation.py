import os
import sys
import torch
import torch.nn as nn
import wandb

from create_dataset import get_dataloaders
from model import build_model, count_params
from evaluate import metrics
from train import train_epoch, validate, config_dic


def train_baseline(model_type: str, dataloaders, device, num_epochs=5):
    """
    Quick training run for ablation baselines.
    model_type: 'frozen_pretrained' or 'random_init'
    """
    if model_type == "frozen_pretrained":
        model = build_model(num_classes=10, freeze_backbone=True)
        print("\nBaseline: Frozen pretrained backbone")
    elif model_type == "random_init":
        model = build_model(num_classes=10, freeze_backbone=False)
        # Re-initialise ALL weights randomly (remove ImageNet pretraining)
        for module in model.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        print("\nBaseline: Random initialisation (no pretrained weights)")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model = model.to(device)
    count_params(model)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3
    )

    for epoch in range(1, num_epochs + 1):
        train_epoch(
            model, dataloaders["train"], criterion, optimizer,
            device, f"{model_type} epoch {epoch}/{num_epochs}"
        )

    val_loss, val_logits, val_labels = validate(
        model, dataloaders["val"], criterion, device
    )
    met = metrics(val_logits, val_labels)
    return met


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dl,test_dl,val_dl,dataset_sizes = get_dataloaders(
        csv_path=config_dic["csv_path"],
        data_dir=config_dic["data_dir"],
        batch_size=config_dic["batch_size"],
        num_workers=config_dic["num_workers"],
    )
    dataloaders={
        "train":train_dl,
        "test":test_dl,
        "val":val_dl
    }

    # Baseline 1: frozen pretrained
    metrics_frozen = train_baseline("frozen_pretrained", dataloaders, device)

    # Baseline 2: random init
    metrics_random = train_baseline("random_init", dataloaders, device)

    # Load your best fine-tuned model
    model = build_model(num_classes=10, freeze_backbone=False)
    ckpt = torch.load("checkpoints/best_model.pth", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    _, fine_logits, fine_labels = validate(model, dataloaders["test"], criterion, device)
    metrics_finetuned = metrics(fine_logits, fine_labels)

    # Print comparison table
    print("\n" + "="*60)
    print("  ABLATION STUDY RESULTS")
    print("="*60)
    print(f"{'Condition':<35} {'mAP':>8} {'F1 Macro':>10}")
    print("-"*60)
    print(f"{'Random init (no pretrained)':<35} {metrics_random['mAP']:>8.4f} {metrics_random['f1_macro']:>10.4f}")
    print(f"{'Frozen pretrained backbone':<35} {metrics_frozen['mAP']:>8.4f} {metrics_frozen['f1_macro']:>10.4f}")
    print(f"{'3-stage fine-tuned (ours)':<35} {metrics_finetuned['mAP']:>8.4f} {metrics_finetuned['f1_macro']:>10.4f}")
    print("="*60)
    print("\nCopy this table into your README.")


if __name__ == "__main__":
    main()