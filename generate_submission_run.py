# generate_submission_run.py
import sys
import os
import torch

# Make sure 'src' is in the path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from model import DeepCustomCNN  # ✅ Your custom CNN
from predict import generate_submission

# ✅ Detect GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# ✅ Load model
model = DeepCustomCNN()
model.load_state_dict(torch.load("outputs/model.pth", map_location=device))
model = model.to(device)

# ✅ Generate predictions
generate_submission(
    model=model,
    test_dir="data/test",
    sample_csv="data/sample_submission.csv",
    output_csv="outputs/submission.csv"
)

