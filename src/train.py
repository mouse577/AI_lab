# src/train.py

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from model import DeepCustomCNN  # 🔁 Make sure you're on custom-deep-cnn branch

# ✅ GPU detection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

def get_dataloaders(train_csv, train_dir, batch_size=64):
    df = pd.read_csv(train_csv)
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'])

    class HistopathologicDataset(torch.utils.data.Dataset):
        def __init__(self, dataframe, dir_path, transform=None):
            self.data = dataframe
            self.dir = dir_path
            self.transform = transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            img_id = self.data.iloc[idx]['id']
            label = self.data.iloc[idx]['label']
            img = Image.open(os.path.join(self.dir, img_id + '.tif'))
            if self.transform:
                img = self.transform(img)
            return img, torch.tensor(label, dtype=torch.float32)

    train_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor()
    ])

    val_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor()
    ])

    train_dataset = HistopathologicDataset(train_df, train_dir, train_transform)
    val_dataset = HistopathologicDataset(val_df, train_dir, val_transform)

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), \
           DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

def train_model(epochs=5, save_path='outputs/model.pth'):
    model = DeepCustomCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # train_loader, _ = get_dataloaders('data/train_labels.csv', 'data/train')
    train_loader, val_loader = get_dataloaders('data/train_labels.csv', 'data/train')

    best_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        start = time.time()

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", dynamic_ncols=True)
        for imgs, labels in progress_bar:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(train_loader)
        tqdm.write(f"📈 Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}")

        end = time.time()
        tqdm.write(f"⏱️ Epoch {epoch+1} took {end - start:.2f} seconds")

        # 🔍 Validation pass
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = model(imgs).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        tqdm.write(f"🔎 Validation Loss: {avg_val_loss:.4f}")

        # 🛑 Early stopping check
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            tqdm.write(f"💾 Saved new best model (val loss: {best_loss:.4f})")
        else:
            patience_counter += 1
            tqdm.write(f"⏸️ No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                tqdm.write("🛑 Early stopping triggered.")
                break

    # return model after loop ends
    return model

    # # ✅ Save model
    # torch.save(model.state_dict(), save_path)
    # print(f"✅ Model saved to: {save_path}")
    # return model

