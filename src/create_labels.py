import os
import numpy as np
import pandas as pd
import random
import csv

random.seed(42)
DATA_DIR="data/EuroSAT"
OUTPUT="data/labels.csv"

CLASSES=os.listdir(DATA_DIR)
CLASS_INDEX_MAP={}
i=0
for cls in CLASSES:
    CLASS_INDEX_MAP[cls]=i
    i=i+1

related_list= {
    "Residential": ["Highway","River"],
    "Industrial": ["Highway"],
    "Highway": ["Residential","Industrial","Pasture"],
    "River": ["HerbaceousVegetation","Residential"],
    "AnnualCrop": ["HerbaceousVegetation"] ,
    "PermanentCrop": ["AnnualCrop","Pasture"]
}

rate=0.15
rows=[]
mult_lab=0
for cls in CLASSES:
    path=os.path.join(DATA_DIR,cls)
    images=sorted(os.listdir(path))
    for img in images:
        img_path=os.path.join(path,img)
        label_vector=[0]*len(CLASSES)
        label_vector[CLASS_INDEX_MAP[cls]]=1

        if cls in related_list and random.random()<rate:
            secondary_class=random.choice(related_list[cls])
            label_vector[CLASS_INDEX_MAP[secondary_class]]=1
            mult_lab=mult_lab+1
        
        rows.append([img_path]+label_vector)

header=["image_path"]+CLASSES

with open(OUTPUT, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print("Length: ",len(rows))
print("Multi Label: ", mult_lab)