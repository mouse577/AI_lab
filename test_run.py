import sys, os
sys.path.append(os.path.join(os.getcwd(), 'src'))

from data_utils import load_labels, show_sample_images, plot_label_distribution
from model import HistopathologicCNN
from train import train_model

print("✅ Loading labels...")
df = load_labels("data/train_labels.csv")
print(df.head())

print("✅ Showing sample images...")
show_sample_images(df, "data/train", num=5)

print("✅ Showing label distribution plot...")
plot_label_distribution(df)

print("✅ Training model on a few batches (quick test)...")
model = train_model(epochs=1, max_batches=5)  # You can set these to small numbers for fast test

print("✅ Test complete! Model trained on small sample.")
