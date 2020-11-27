import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

random.seed(100)

dataset = pd.read_csv("datasets/original/train.csv")

dataset = dataset.dropna(subset=["text"])

print(dataset.describe)

_ , df = train_test_split(df,test_size=0.2,stratify=df["sentiment"])

df.to_csv("datasets/original/dataset.csv", sep=";")
