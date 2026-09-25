# Histopathologic Cancer Detection with PyTorch

A research and coursework project for binary classification of histopathology image patches using a custom convolutional neural network (CNN).

The repository contains PyTorch code for loading labeled images, training and validating a CNN, saving the best checkpoint, and generating prediction probabilities in a submission CSV.

## What is implemented

- A five-block convolutional neural network in `src/model.py`
- Image resizing and training augmentation
- Stratified training/validation split
- Binary classification using `BCEWithLogitsLoss`
- Validation-loss tracking and early stopping
- Model checkpoint saving and prediction export

## Repository guide

| File | Purpose |
| --- | --- |
| `src/model.py` | Defines the custom CNN |
| `src/train.py` | Loads images, trains the model, and evaluates validation loss |
| `src/predict.py` | Generates prediction probabilities for test images |
| `train_run.py` | Starts training |
| `generate_submission_run.py` | Loads a saved model and generates a submission CSV |
| `train_labels.csv` | Training labels |
| `sample_submission.csv` | Example submission format |
| `Figure_1.png` | Project figure |
| `requirements.txt` | Python dependencies |

## Data and reproducibility

The histopathology image files and a trained model checkpoint are not included in this repository. The scripts expect image files and CSVs in a local `data/` directory:

```text
data/
├── train/
├── test/
├── train_labels.csv
└── sample_submission.csv
```

Before running, place the CSVs and appropriately licensed image data at those paths and create an `outputs/` directory for the checkpoint and submission. The CSV files currently at the repository root can be copied into `data/`.

The training script resizes images to 96 × 96 pixels, uses an 80/20 stratified split, and trains for five epochs by default. Run the scripts from the repository root:

```bash
python train_run.py
python generate_submission_run.py
```

The prediction script requires the saved checkpoint at `outputs/model.pth`.

## Project report

The repository currently contains an empty `HW5_Report.pdf` placeholder. The report should be uploaded before it is presented as documentation or results.
