import pandas as pd
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

mu=[0.485,0.456,0.406]
sigma=[0.229,0.224,0.225]


train_transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(
                brightness=0.2, contrast=0.2,
                saturation=0.2, hue=0.05
            ),
    transforms.ToTensor(),
    transforms.Normalize(mean=mu,std=sigma)
])

tv_transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mu,std=sigma)
])

class CustomDataset(Dataset):
    def __init__(self,df,data_dir,transform,class_names):
        self.df=df.reset_index(drop=True)
        self.data_dir=data_dir
        self.transform=transform
        self.class_names=class_names
    

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,index):
        row=self.df.iloc[index]
        img_path=row["image_path"]

        image=Image.open(img_path).convert("RGB")
        image=self.transform(image)

        labels=torch.tensor(row[self.class_names].values.astype(np.float32),dtype=torch.float32)

        return image,labels
    
def get_dataloaders(csv_path: str,
                    data_dir: str,
                    batch_size: int =32,
                    num_workers: int=4,
                    val_size: float=0.15,
                    test_size: float=0.15,
                    random_state: int=42):
    df=pd.read_csv(csv_path)
    class_names=os.listdir(os.path.join(data_dir,"EuroSAT"))
    tv=test_size+val_size
    tvrat=test_size/tv
    train_df,temp_df=train_test_split(df, test_size=tv, random_state=random_state, shuffle=True)
    test_df,val_df=train_test_split(temp_df, test_size=tvrat, random_state=random_state, shuffle=True)

    train_ds=CustomDataset(train_df,data_dir,train_transform,class_names)
    test_ds=CustomDataset(test_df,data_dir,tv_transform,class_names)
    val_ds=CustomDataset(val_df,data_dir,tv_transform,class_names)

    train_dl=DataLoader(train_ds,num_workers=num_workers,batch_size=batch_size,shuffle=True,pin_memory=True)

    test_dl=DataLoader(test_ds,num_workers=num_workers,batch_size=batch_size,shuffle=False,pin_memory=True)

    val_dl=DataLoader(val_ds,num_workers=num_workers,batch_size=batch_size,shuffle=False,pin_memory=True)

    dataset_sizes={"train":len(train_ds),
                   "test": len(test_ds),
                   "val": len(val_ds)}

    return train_dl,test_dl,val_dl,dataset_sizes


if __name__ =="__main__":
    train,test,val,sizes=get_dataloaders(
        csv_path='data/labels.csv',
        data_dir='data',
        batch_size=8,
        num_workers=4
    )

    images, labels = next(iter(train))
    print(f"Image batch shape : {images.shape}")   
    print(f"Label batch shape : {labels.shape}")   # [8, 10]
    print(sizes)



