import torch
import torch.nn as nn
import torchvision.models as models
from torchinfo import summary

def build_model(num_classes:int=10, freeze_backbone: bool=True):
    weights=models.ResNet50_Weights.IMAGENET1K_V2
    model=models.resnet50(weights=weights)

    for param in model.parameters():
        param.requires_grad=False

    in_features=model.fc.in_features
    model.fc=nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features,num_classes)
    )

    for param in model.fc.parameters():
        param.requires_grad = True

    if not freeze_backbone:
        for param in model.parameters():
            param.requires_grad = True

    return model

def unfreeze_layer(model,layer_name: str):
    layer=getattr(model,layer_name, None)
    if layer is None:
        raise ValueError("Layer not found in model")
    for param in layer.parameters():
        param.requires_grad=True
    print("Layer unfrozen")

def count_params(model):
    trainable=0
    total=0
    for p in model.parameters():
        total=total+p.numel()
        if p.requires_grad:
            trainable=trainable+p.numel()
    print(f"Total parameters: {total:,}" )
    print(f"Trainable parameters: {trainable:,}" )
    print(f"Frozen parameters: {total-trainable:,}" )

if __name__ == "__main__":
    model = build_model(num_classes=10, freeze_backbone=True)
    count_params(model)

    
    dummy_input = torch.randn(4, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}") 
    summary(model,input_size=dummy_input.shape)
 