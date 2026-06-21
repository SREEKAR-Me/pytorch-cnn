import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb

from create_dataset import get_dataloaders
from model import build_model, unfreeze_layer, count_params
from evaluate import metrics

config_dic = {
    "csv_path"     : "data/labels.csv",
    "data_dir"     : "data",
    "num_classes"  : 10,
    "batch_size"   : 32,
    "num_workers"  : 4,

    # Stage 1 — Training head only
    "stage1_epochs": 5,
    "stage1_lr"    : 1e-3,

    # Stage 2 — Only Partial unfreeze (layer3 + layer4)
    "stage2_epochs": 8,
    "stage2_lr"    : 1e-4,

    # Stage 3 — Complete fine-tune of the model
    "stage3_epochs": 15,
    "stage3_lr"    : 1e-5,

    # Store and replicate
    "checkpoint_dir": "checkpoints",
    "seed"           : 42,
}

def set_seed(seed):
    #For replication of results
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_epoch(model, loader, criterion, optimizer, device, epoch_name):
    model.train()
    avg_loss=0.0

    progress=tqdm(loader,desc=epoch_name,leave=False)

    for images, labels in progress:
        images=images.to(device)
        labels=labels.to(device)

        optimizer.zero_grad()
        pred=model(images)
        loss=criterion(pred,labels)
        loss.backward()
        optimizer.step()

        avg_loss=avg_loss+loss.item()*images.size(0)
    avg_loss=avg_loss/len(loader.dataset)

    return avg_loss

def validate(model,loader,criterion,device):
    model.eval()
    curr_loss=0.0
    all_preds=[]
    all_labels=[]

    with torch.no_grad():
        for images,labels in loader:
            images=images.to(device)
            labels=labels.to(device)
            preds=model(images)
            loss=criterion(preds,labels)
            curr_loss=curr_loss+loss.item()*images.size(0)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds=torch.cat(all_preds)
    all_labels=torch.cat(all_labels)

    val_loss=curr_loss/len(loader.dataset)
    return val_loss, all_preds,all_labels

def run_stage(stage_num,model,dataloaders,criterion,optimizer,scheduler,num_epochs,device,best_map):
    print(f"Stage {stage_num}")
    print(f"Number of Epochs: {num_epochs}")
    count_params(model)

    for epoch in range(1,num_epochs+1):
        epoch_name = f"Stage Name: {stage_num}  Epoch: {epoch}/{num_epochs}"
        train_loss=train_epoch(model,dataloaders["train"],criterion,optimizer,device,epoch_name)
        val_loss,val_preds,val_labels=validate(model,dataloaders["val"],criterion,device)
        comp=metrics(val_preds,val_labels)

        if scheduler:
            scheduler.step()

        wandb.log({"stage" : stage_num,
                   "epoch": epoch,
                   "Train Loss": train_loss,
                   "Validation Loss": val_loss,
                   "Validation mAP": comp["mAP"],
                   "Validation F1 Macro": comp["f1_macro"],
                   "Learning Rate": optimizer.param_groups[0]["lr"]})
        
        print(
            f"  [{epoch_name}] "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"mAP={comp['mAP']:.4f} | "
            f"F1={comp['f1_macro']:.4f}"
        )

        if comp["mAP"] > best_map:
            best_map = comp["mAP"]
            ckpt_path = os.path.join(config_dic["checkpoint_dir"], "best_model.pth")
            torch.save({
                "stage"     : stage_num,
                "epoch"     : epoch,
                "model_state_dict" : model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_map"  : best_map,
            }, ckpt_path)
            print(f"Checkpoint saved.")

    return best_map


def main():
    set_seed(config_dic["seed"])
    os.makedirs(config_dic["checkpoint_dir"],exist_ok=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    wandb.init(project="EuroSAT Project",config=config_dic,name="resnet50-finetuning")


    train_dl,test_dl,val_dl,sizes=get_dataloaders(
        csv_path=config_dic["csv_path"],
        data_dir=config_dic["data_dir"],
        batch_size=config_dic["batch_size"],
        num_workers=config_dic["num_workers"]
    )
    dataloaders={
        "train":train_dl,
        "test":test_dl,
        "val":val_dl
    }
    
    criterion=nn.BCEWithLogitsLoss()
    best_map=0.0

    model=build_model(num_classes=config_dic["num_classes"],freeze_backbone=True)
    model=model.to(device)

    optimizer=optim.Adam(
        filter(lambda p:p.requires_grad,model.parameters()),
        lr=config_dic["stage1_lr"]
    )

    best_map=run_stage(stage_num=1,model=model,dataloaders=dataloaders,criterion=criterion,optimizer=optimizer,scheduler=None,num_epochs=config_dic["stage1_epochs"],device=device,best_map=best_map)


    #STAGE 2: UNFREEZE LAYERS 3 AND 4
    unfreeze_layer(model,"layer3")
    unfreeze_layer(model,"layer4")

    optimizer=optim.Adam(
        filter(lambda p: p.requires_grad,model.parameters()),
        lr=config_dic["stage2_lr"]
    )
 
    best_map=run_stage(stage_num=2,model=model,dataloaders=dataloaders,criterion=criterion,optimizer=optimizer,scheduler=None,num_epochs=config_dic["stage2_epochs"],device=device,best_map=best_map)


    #STAGE 3: FULL FINETUNING WITH COSINE ANNEALING
    for p in model.parameters():
        p.requires_grad=True

    optimizer=optim.Adam(
        model.parameters(),
        lr=config_dic["stage3_lr"]
    )

    scheduler=CosineAnnealingLR(optimizer, T_max=config_dic["stage3_epochs"], eta_min=1e-7)
 
    best_map=run_stage(stage_num=3,model=model,dataloaders=dataloaders,criterion=criterion,optimizer=optimizer,scheduler=scheduler,num_epochs=config_dic["stage3_epochs"],device=device,best_map=best_map)

    print(f"\nTraining complete. Best mAP: {best_map:.4f}")
    wandb.finish()

if __name__ == "__main__":
    main()

    


