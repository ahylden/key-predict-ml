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

Steps 1-3 are optional as pre trained models are included in ./models

1. Download dataset
2. Create Augmented dataset
    - python key_shift.py
3. Train model
    - python key_finder.py (default is 100 epochs, use --epochs to change)
4. Find Keys
    - python key_finder.py --find-key (use either --song or --batch)

Optional Arguments:

--find-key
- Run model on a single audio or whole directory file to find its key

--song
- Path to the audio file to analyze when using --find-key (ex: ./songs/song.mp3)

--batch
- Path to the directory with audio files to analyze when using --find-key (ex ./songs)
- supported file types are .mp3 and .wav

--test-batch
- Test model on the full dataset and log predictions vs expected

--epochs
- Number of training epochs (default= 100) When running training sweep this should be multiple of 10

--load-model
- Path to a saved model, default is best CQT model from training. When using a model of non CQT type, specify --logmel or --chroma when running

--data-size
- Limit the number of samples used for training/testing

--cqt
- This is the default type and does not necessarily need to be specified
- Use Constant Q Transform (CQT) spectrograms. Used in training or key finding.

--logmel
- Use logmel features instead of CQT spectrograms. Used in training or key finding.. Used in training or key finding.

--chroma
- Use chroma features instead of CQT spectrograms. Used in training or key finding.

--build-cache
- Precompute and store audio features for selected feature type (default= logmel if no other arg specified) for faster training/testing

--run-training-sweep
- Run sweep of trainings with different feature types (log-mel, cqt, chroma) and different epoch lengths

---

## Code

Written by me:
- Model architecture (modeled after reference paper)
- Training loop
- Dataset pipeline (modeled after reference paper)
- Feature extraction and caching
- Augmentation integration
- Visualization tools

Adapted concepts:
- PyTorch training workflows (adapted from lecture slides/examples)
- Librosa feature extraction (adapted from librosa library examples)

External libraries:
- Librosa
- PyTorch
- scikit-learn

No external code copied.
