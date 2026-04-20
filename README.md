# Key Classification using CNN

## Overview
This project implements a deep learning pipeline for musical key classification using PyTorch. The model predicts one of 24 classes (12 major + 12 minor keys) from audio signals using time-frequency representations. The pipeline is based on a paper by Filip Korzeniowski and Gerhard Widme (https://arxiv.org/pdf/1706.02921)

Features include:
- Log-Mel Spectrograms (default)
- Constant Q Transform (CQT)
- Chroma Features
- Dataset augmentation via pitch shifting
- Training, evaluation, and visualization tools

---

## Project Structure

- key_finder.py
- key_shift.py
- models/
- feature_cache/
- training_plots/
- test_plots/
- giantsteps-key-dataset/
- augmented-data/
- README.md

---

## Dependencies

pip install numpy matplotlib librosa torch scikit-learn

sudo apt install sox

- Python ≥ 3.8
- SoX (required for pitch shifted dataset)


---

## Dataset Setup (GiantSteps)

### Clone dataset

git clone https://github.com/GiantSteps/giantsteps-key-dataset.git

### Download audio

bash audio_dl.sh

This script downloads the audio files.

---

## Dataset Augmentation

Run:

python key_shift.py

- Applies pitch shifts [-3, -2, -1, +1, +2, +3]
- Updates key labels to match new key
- Outputs to ./augmented-data/

---

## How to Run Model

Train:
- python key_finder.py

Optional Arguments:

--predict-key
- Run model on a single audio file to predict its key

--song
- Path to the audio file to analyze when using --predict-key

--test-batch

- Test model on the full dataset and log predictions vs expected
--epochsb
- Number of training epochs (default= 10)

--load-model
- Path to a saved model, default is best model from training

--data-size
- Limit the number of samples used for training/testing

--cqt
- Use Constant Q Transform (CQT) features instead of log-mel spectrograms

--chroma
- Use chroma features instead of log-mel spectrograms

--build-cache
- Precompute and store audio features for selected feature type (default= logmel if no other arg specified) for faster training/testing

--run-training-sweep
- Run sweep of trainings with different feature types (log-mel, cqt, chroma) and different epoch lengths

---

## Code

Written by me:
- Model architecture
- Training loop
- Dataset pipeline
- Feature extraction + caching
- Augmentation integration
- Visualization tools

Adapted concepts:
- PyTorch training workflows
- Librosa feature extraction

External libraries:
- Librosa
- PyTorch
- scikit-learn

No external code copied.
