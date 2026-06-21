import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score,average_precision_score, precision_recall_curve


threshold=0.5

def metrics(logits: torch.Tensor, labels: torch.Tensor):
    if isinstance(logits, torch.Tensor):
        logits_np = logits.detach().numpy()
    else:
        logits_np = np.array(logits)

    if isinstance(labels, torch.Tensor):
        labels_np = labels.detach().numpy()
    else:
        labels_np = np.array(labels)
 
    labels_int = labels_np.astype(int)

  
    probs = 1.0 / (1.0 + np.exp(-logits_np))

    preds = (probs >= threshold).astype(int)


    ap_per_class = []
    valid_ap = []
    for i in range(labels_int.shape[1]):
        if labels_int[:, i].sum() > 0:
            ap = float(average_precision_score(labels_int[:, i], probs[:, i]))
            valid_ap.append(ap)
        else:
            ap = 0.0
        ap_per_class.append(ap)

    mAP = float(np.mean(valid_ap)) if valid_ap else 0.0

        
    if labels_int.ndim == 1:
        labels_int = labels_int.reshape(-1, 1)
        preds = preds.reshape(-1, 1)

    f1_per_class = f1_score(labels_int, preds, average=None, zero_division=0)
    f1_macro = f1_score(labels_int, preds, average="macro", zero_division=0)

    return {
        "f1_macro"    : float(f1_macro),
        "f1_per_class": f1_per_class.tolist(),
        "mAP"         : mAP,
        "ap_per_class": ap_per_class,
        "probs"       : probs,
        "labels_int"  : labels_int,
    }
    


def evaluate_on_test(model, dataloader, criterion, class_names, device):
    model.eval()
    curr_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            preds = model(images)
            loss = criterion(preds, labels)
            curr_loss += loss.item() * images.size(0)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    test_loss = curr_loss / len(dataloader.dataset)

    comp = metrics(all_logits, all_labels)

    print(f"  Loss          : {test_loss:.4f}")
    print(f"  Macro F1      : {comp['f1_macro']:.4f}")
    print(f"  mAP           : {comp['mAP']:.4f}")
    print("\n  Per-class F1:")
    for cls, f1 in zip(class_names, comp["f1_per_class"]):
        print(f"    {cls:<25}: {f1:.4f}")

    return comp