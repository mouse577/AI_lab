# src/data_utils.py

import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image

def load_labels(csv_path):
    df = pd.read_csv(csv_path)
    return df

import matplotlib.pyplot as plt
from PIL import Image
import os

def show_sample_images(df, train_dir, num=5):
    fig, axs = plt.subplots(1, num, figsize=(15, 5))
    for i in range(num):
        img_id = df.iloc[i]['id']
        label = df.iloc[i]['label']
        img_path = os.path.join(train_dir, img_id + '.tif')
        try:
            img = Image.open(img_path)
            axs[i].imshow(img)
            axs[i].set_title(f"Label: {label}")
            axs[i].axis('off')
        except FileNotFoundError:
            axs[i].set_title("Missing Image")
            axs[i].axis('off')
    plt.tight_layout()
    plt.show()  # 👈 this is critical to force plot display in scripts


def plot_label_distribution(df):
    df['label'].value_counts().plot(kind='bar', title='Label Distribution')
    plt.xticks([0, 1], ['No Tumor', 'Tumor'])
    plt.show()
