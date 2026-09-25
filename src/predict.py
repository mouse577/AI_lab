import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from model import DeepCustomCNN  # ✅ Update this for your custom model

def generate_submission(model, test_dir, sample_csv, output_csv='outputs/submission.csv'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device for prediction: {device}")

    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor()
    ])

    df = pd.read_csv(sample_csv)
    predictions = []

    with torch.no_grad():
        for img_id in df['id']:
            img_path = os.path.join(test_dir, f"{img_id}.tif")
            try:
                img = Image.open(img_path).convert("RGB")
                img = transform(img).unsqueeze(0).to(device)  # Shape: (1, 3, 96, 96)

                # ✅ Apply sigmoid for binary probability output
                output = model(img)
                prob = float(torch.sigmoid(output).squeeze().item())

            except Exception as e:
                print(f"❌ Failed to process {img_id}: {e}")
                prob = 0.0  # fallback

            predictions.append(prob)

    df['label'] = predictions
    df.to_csv(output_csv, index=False)
    print(f"✅ Submission file saved to: {output_csv}")




