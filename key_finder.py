import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import sys
import os
import json
import argparse
from glob import glob
from pathlib import Path
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()

parser.add_argument("--find-key", action="store_true", help="Run model on a single audio file or whole directory to find its key")
parser.add_argument("--song", default=None, help="Path to the audio file to analyze when using --find-key")
parser.add_argument("--batch", default=None, help="Path to the directory with audio files to analyze when using --find-key")
parser.add_argument("--test-batch", action="store_true", help="Test model on the full dataset and log predictions vs expected")
parser.add_argument("--epochs", default=100, help="Number of training epochs (default= 100). When running training sweep this should be multiple of 10")
parser.add_argument("--load-model", default=None, help="Path to a saved model, default is best model from training")
parser.add_argument("--data-size", default=None, help="Limit the number of samples used for training/testing")
parser.add_argument("--cqt", action="store_true", help="Use Constant Q Transform (CQT) spectrograms")
parser.add_argument("--chroma", action="store_true", help="Use chroma spectrograms")
parser.add_argument("--logmel", action="store_true", help="Use log-mel spectrograms")
parser.add_argument("--build-cache", action="store_true", help="Precompute and cache audio features for faster training/testing")
parser.add_argument("--run-training-sweep", action="store_true", help="Run sweep of trainings with different feature types (log-mel, cqt, chroma) and different epoch lengths")

args = parser.parse_args()

key_class = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

torch.cuda.empty_cache()

def get_cache_path(audio_path, spec_type, target_length=600):
    base = os.path.splitext(os.path.basename(audio_path))[0]
    cache_dir = os.path.join("feature_cache", spec_type)
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{base}_{target_length}.npy")

def build_cache(root):
    data = []

    data.extend(load_gs_dataset(root))

    if args.data_size is not None:
        data = data[:int(args.data_size)]

    spec_type = get_spec_type()

    for index, item in enumerate(data):
        _ = load_feature_cached(item["path"], spec_type)
        print(f"Cached {index+1}/{len(data)}: {item['path']}")

def load_feature_cached(audio_path, spec_type):
    cache_path = get_cache_path(audio_path, spec_type)

    if os.path.exists(cache_path):
        return np.load(cache_path)

    if spec_type == "chroma":
        feature = chroma_spec(audio_path)
    elif spec_type == "cqt":
        feature = cqt_log_spec(audio_path)
    else:
        feature = logmel(audio_path, n_fft=8192, n_mels=105, hop_length=8820)

    np.save(cache_path, feature.astype(np.float32))
    return feature.astype(np.float32)

def json_key_decode(key_value, mode_value):
    enharmonic_map = {
        "c": "C",
        "b#": "C",
        "c#": "Db",
        "db": "Db",
        "d": "D",
        "d#": "Eb",
        "eb": "Eb",
        "e": "E",
        "fb": "E",
        "f": "F",
        "e#": "F",
        "f#": "Gb",
        "gb": "Gb",
        "g": "G",
        "g#": "Ab",
        "ab": "Ab",
        "a": "A",
        "a#": "Bb",
        "bb": "Bb",
        "b": "B",
        "cb": "B"
    }

    key_value = str(key_value).strip().lower()
    mode_value = str(mode_value).strip().lower()

    if key_value not in enharmonic_map:
        raise ValueError(f"Unknown FS key: {key_value}")

    tonic = enharmonic_map[key_value]

    if mode_value in ("maj", "major"):
        return key_class.index(tonic)
    elif mode_value in ("min", "minor"):
        return key_class.index(tonic) + 12
    else:
        raise ValueError(f"Unknown FS mode: {mode_value}")

def load_gs_dataset(root):
    audio_dir = os.path.join(root, "audio")
    data = []
    key_files = glob(os.path.join(root, "annotations", "key", "*.key"))

    if args.data_size != None:
        key_files = key_files[:int(args.data_size)]

    for key in key_files:
        name = os.path.splitext(os.path.basename(key))[0]
        with open(key, "r", encoding="utf-8") as f:
            meta = f.read().strip()
        key_label = key_decode(meta)
        audio_path = os.path.join(audio_dir, name + ".wav")
        data.append({
            "path": audio_path,
            "key": key_label
        })

    return data

def load_dataset(root, spec_type):
    data = []

    data.extend(load_gs_dataset(root))

    if args.data_size is not None:
        data = data[:int(args.data_size)]

    X_list = []
    y = []

    for index, track in enumerate(data):
        X_list.append(load_feature_cached(track["path"], spec_type))
        y.append(track["key"])
        print(f"Loaded Training Data {index+1}/{len(data)}")

    X = np.stack(X_list)
    return X, y

def key_decode(giantsteps_key):
    tonic, mode = giantsteps_key.strip().split(" ")

    if mode.lower() == "major":
        return key_class.index(tonic)
    if mode.lower() == "minor":
        return key_class.index(tonic) + 12
    
def key_return(index):
    if index > 11:
        return f"{key_class[index-12]} minor"
    return f"{key_class[index]} major"

def decode_key_rel(k):
    if k < 12:
        return k, "major"
    return (k - 12), "minor"

def is_relative(pred, actual):
    pred_i, pred_mode = decode_key_rel(pred)
    actual_i, actual_mode = decode_key_rel(actual)

    if pred_mode == actual_mode:
        return False
    elif actual_mode == "major":
        rel_minor = (actual_i - 3) % 12
        return pred_mode == "minor" and pred_i == rel_minor
    else:
        rel_major = (actual_i + 3) % 12
        return pred_mode == "major" and pred_i == rel_major

def cqt_log_spec(file_path, target_length=600):
    y, sr = librosa.load(file_path)
    C = np.abs(librosa.cqt(y, sr=sr, bins_per_octave=24, n_bins=120, hop_length=sr//5))
    spec = np.log1p(C)

    F, T = spec.shape
    if T < target_length:
        spec = np.pad(spec, ((0,0),(0,target_length-T)))
    else:
        start = (T - target_length)//2
        spec = spec[:, start:start+target_length]

    return spec

def chroma_spec(file_path, target_length=600, hop_length=4096):
    y, sr = librosa.load(file_path, sr=44100)

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length, bins_per_octave=24)

    chroma = chroma.astype(np.float32)

    T = chroma.shape[1]
    if T < target_length:
        chroma = np.pad(chroma, ((0, 0), (0, target_length - T)))
    else:
        start = (T - target_length) // 2   # deterministic crop
        chroma = chroma[:, start:start + target_length]

    return chroma

def logmel(file_path, n_fft, n_mels, hop_length, target_length=600, plot=False):
    y, sr = librosa.load(file_path, sr=44100)
    spectogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, n_mels=n_mels, hop_length=hop_length)
    spec_db = librosa.power_to_db(spectogram, ref=np.max)

    length = spec_db.shape[1]

    if length < target_length:
        pad = target_length - length
        spec_db = np.pad(spec_db, ((0, 0), (0, pad)))
    elif length > target_length:
        start = np.random.randint(0, length-target_length)
        spec_db = spec_db[:, start:start+target_length]

    if plot:
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(spec_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="mel")
        plt.colorbar(format="%+2.0f dB")
        plt.title("Mel Spectrogram")
        plt.tight_layout()
        plt.show()

    return spec_db

# ----------- CNN Model -----------

class FeatureExtractor(nn.Module):
    def __init__(self, spec_type):
        super().__init__()

        if spec_type == "chroma":
            self.bins = 12
        elif spec_type == "cqt":
            self.bins = 120
        else:
            self.bins = 105

        self.conv1 = nn.Conv2d(1, 8, 5, padding=2)
        self.conv2 = nn.Conv2d(8, 8, 5, padding=2)
        self.conv3 = nn.Conv2d(8, 8, 5, padding=2)
        self.conv4 = nn.Conv2d(8, 8, 5, padding=2)
        self.conv5 = nn.Conv2d(8, 8, 5, padding=2)

        self.dropout2d = nn.Dropout2d(0.4)

        self.freq_dense = nn.Linear(8*self.bins, 48)

        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(48, 24)

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = F.elu(self.conv5(x))

        x = self.dropout2d(x)

        x = x.permute(0, 3, 1, 2)
        x = x.reshape(x.size(0), x.size(1), -1)
        x = F.elu(self.freq_dense(x))
        x = x.mean(dim=1)
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x
    
# ----------- Training Loop -----------

def test_model(model, dataloader, loss_fn, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device, dtype=torch.float32)
            if xb.dim() == 3:
                xb = xb.unsqueeze(1)
            yb = yb.to(device).long()

            output = model(xb)
            loss = loss_fn(output, yb)

            running_loss += loss.item() * xb.size(0)
            preds = output.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy
    
def train_model(model, dataloader_train, dataloader_val, optimizer, loss_fn, device, epochs=20, model_path="best_key_model.pt"):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10, threshold=0.010, min_lr=1e-5)

    best_val_acc = 0.0
    best_model_path = f"models/{model_path}"

    training_data = []

    for epoch in range(1, epochs + 1):
        model.train()

        train_loss = []
        for batch_idx, (xb, yb) in enumerate(dataloader_train):
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            output = model(xb)
            loss = loss_fn(output, yb)

            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        val_loss, val_acc = test_model(model, dataloader_val, loss_fn, device)

        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_acc)
        new_lr = optimizer.param_groups[0]["lr"]

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with val acc = {best_val_acc:.4f}")

        if new_lr < old_lr:
            print(f"Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")

        print(f"Epoch {epoch}/{epochs} | Train Loss: {np.mean(train_loss):.4f} || Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} || LR: {new_lr:.6f}")
        training_data.append((np.mean(train_loss), val_loss, val_acc))

    return training_data

def test_batch(model, device, spec_type, epochs=args.epochs, load_model=args.load_model, show=True):
    correct = 0
    total = 0
    relative = 0

    model.load_state_dict(torch.load(load_model))
    model.to(device)
    x, y = load_dataset("./giantsteps-key-dataset", spec_type=spec_type)
    x_tensor = torch.tensor(x).unsqueeze(1).to(device)
    preds = model(x_tensor).argmax(dim=1)
    misses = np.zeros(24)
    for pred, yb in zip(preds, y):
        print(f"Predicted Key: {key_return(pred)} | Actual Key: {key_return(yb)}")
        if pred == yb:
            correct += 1
        elif is_relative(pred, yb):
            relative += 1
            misses[yb] += 1
        else:
            misses[yb] += 1
        total += 1
    print(f"{correct}/{total}")
    print(f"Relative Majors/Minors predicted instead: {relative}")
    print(f"{(correct/total*100):.1f}% Accuracy")
    print(f"Accuracy counting relatives as correct: {((correct+relative)/total*100):.1f}%")
    plot_misses(misses, spec_type, epochs=epochs, show=show)
    return (correct/total*100), ((correct+relative)/total*100)
    
def plot_misses(misses, spec_type, save_dir="test_plots", epochs=args.epochs, show=True):
    labels = [f"{k} maj" for k in key_class] + [f"{k} min" for k in key_class]

    save_path = os.path.join(save_dir, f"{spec_type}_misses_by_key_{epochs}_epochs.png")

    plt.figure(figsize=(14, 5))
    plt.bar(range(24), misses)
    plt.xticks(range(24), labels, rotation=45, ha="right")
    plt.xlabel("Actual Key")
    plt.ylabel("Number of Misses")
    plt.title(f"Model Misses by Actual Key (Total Misses: {int(sum(misses))}/604)")
    plt.ylim(0, 30)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close()


def plot_train_results(train_results, spec_type, epochs=args.epochs, save_dir="training_plots", overlay=False, show=False):

    plt.figure(figsize=(8, 5))

    if overlay:
        if len(train_results) == 3 and all(isinstance(x, list) for x in train_results):
            save_path = os.path.join(save_dir, f"final_overlay.png")
            spec_types = ["chroma", "cqt", "logmel"]
            colors = ['r', 'g', 'b']
            for idx, results in enumerate(train_results):
                epoch = [x[0] for x in results]
                accuracy = [x[1] for x in results]
                accuracy_rel = [x[2] for x in results]
                plt.plot(epoch, accuracy, linestyle="-", marker="o", markersize=3, color=colors[idx], label=f"{spec_types[idx]} Acc")
                plt.plot(epoch, accuracy_rel, linestyle="--", marker="o", markersize=3, color=colors[idx], label=f"{spec_types[idx]} Rel Acc")
            plt.subplots_adjust(right=0.8, left=0.07)
            plt.legend(loc='center left', bbox_to_anchor=(1, .5))
            plt.title(f"Test Accuracy Comparison")
        else:
            epoch = [x[0] for x in train_results]
            accuracy = [x[1] for x in train_results]
            accuracy_rel = [x[2] for x in train_results]
            save_path = os.path.join(save_dir, f"{spec_type}_overlay.png")
            plt.plot(epoch, accuracy, linestyle="-", marker="o", markersize=3, label=f"Test Accuracy")
            plt.plot(epoch, accuracy_rel, linestyle="--", marker="o", markersize=3, label=f"Test Accuracy w/ Relative Maj/Min Accepted")
            plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0))
            plt.title(f"Test Accuracy ({spec_type})")
        plt.xlabel("Epoch")
        plt.ylabel("Test Accuracy")
    else:
        save_path = os.path.join(save_dir, f"{spec_type}_{epochs}_epochs.png")
        train_losses = [x[0] for x in train_results]
        val_losses = [x[1] for x in train_results]
        val_acc = [x[2] for x in train_results]
        epochs = range(1, len(train_results) + 1)
        plt.plot(epochs, train_losses, marker='o', label="Train Loss")
        plt.plot(epochs, val_losses, marker='o', label="Validation Loss")
        plt.plot(epochs, val_acc, marker='o', label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training and Validation Loss ({spec_type})")
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0))

    plt.grid(True)

    if show:
        plt.show()
    else:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        plt.close()

def build_dataloaders(spec_type):
    X, y = load_dataset("./augmented-data", spec_type)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )

    X_tensor_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_tensor_train = torch.tensor(y_train, dtype=torch.long)

    X_tensor_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
    y_tensor_val = torch.tensor(y_val, dtype=torch.long)

    X_tensor_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    y_tensor_test = torch.tensor(y_test, dtype=torch.long)

    dataset_train = TensorDataset(X_tensor_train, y_tensor_train)
    dataset_val = TensorDataset(X_tensor_val, y_tensor_val)
    dataset_test = TensorDataset(X_tensor_test, y_tensor_test)

    dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=32, shuffle=False)
    dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=False)

    return dataloader_train, dataloader_val, dataloader_test

def get_spec_type():
    if args.chroma:
        spec_type = "chroma"
    elif args.logmel:
        spec_type = "logmel"
    else:
        spec_type = "cqt"
    return spec_type

def run_training_sweep():
    epochs = int(args.epochs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spec_types = ["chroma", "cqt", "logmel"]
    comp_results = []

    for spec_type in spec_types:
        print("\n" + "=" * 60)
        print(f"Starting training run: feature={spec_type}, epochs={epochs}")
        print("=" * 60)

        epochs_left = epochs
        epochs_run = 0

        dataloader_train, dataloader_val, dataloader_test = build_dataloaders(spec_type)

        model = FeatureExtractor(spec_type).to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

        results = []
        acc_results = []

        while epochs_left > 0:
            epochs_run += 10
            results.extend(train_model(model, dataloader_train, dataloader_val, optimizer, criterion, device, epochs=10, model_path=f"best_{spec_type}_model_{epochs_run}.pt"))

            avg_loss, accuracy = test_model(model, dataloader_test, criterion, device)
            print(f"Test Loss: {avg_loss:.4f} | Test Accuracy: {accuracy:.4f}")

            perfect_acc, rel_acc = test_batch(model, device, spec_type=spec_type, epochs=epochs_run, load_model=f"./models/best_{spec_type}_model_{epochs_run}.pt", show=False)

            acc_results.append((epochs_run, perfect_acc, rel_acc))
            epochs_left -= 10

        comp_results.append(acc_results)
        plot_train_results(results, spec_type=spec_type, epochs=epochs_run, save_dir="training_plots")
        plot_train_results(acc_results, spec_type=spec_type, epochs=epochs_run, save_dir="training_plots", overlay=True)
    plot_train_results(comp_results, spec_type=get_spec_type(), epochs=epochs_run, save_dir="training_plots", overlay=True)  

def load_model(model, device, model_type):
    if args.load_model != None:
        if not os.path.exists(args.load_model):
            print("Could not find existing model, please train model first")
            sys.exit()
        model.load_state_dict(torch.load(args.load_model))
    else:
        model.load_state_dict(torch.load(f"./models/best_{model_type}_model_100.pt"))
    model.to(device)

def find_key(model, device, song):
    if not os.path.exists(song):
            print(f"Could not find song at file path {song}")
            sys.exit()
    if args.chroma:
        spectrogram = chroma_spec(song)
    elif args.logmel:
        spectrogram = logmel(song, n_fft=8192, n_mels=105, hop_length=8820)
    else:
        spectrogram = cqt_log_spec(song)
    x_tensor = torch.tensor(spectrogram)[None, None, :, :].to(device)
    with torch.no_grad():
        pred = model(x_tensor).argmax(dim=1)
        key = key_return(pred)
        print(f"Predicted Key for song {song}: {key}")
    
    key = key.replace(" ", "_")
    file = Path(song)
    rename = file.with_name(f"{file.stem}_{key}{file.suffix}")
    file.rename(rename)


# ----------- Classifier -----------
model = FeatureExtractor(get_spec_type())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.build_cache:
    print("Building feature cache")
    build_cache("gs", "./augmented-data")
    print("Cache build complete")
    sys.exit()

if(args.find_key):
    load_model(model, device, get_spec_type())
    if args.song != None:
        find_key(model, device, args.song)
    elif args.batch != None:
        if not os.path.exists(args.batch):
            print(f"Could not directory at path {args.batch}")
            sys.exit()
        songs = glob(os.path.join(args.batch, "*.wav")) + glob(os.path.join(args.batch, "*.mp3"))
        for song in songs:
            find_key(model, device, song)
        
elif(args.test_batch):
    if not os.path.exists(args.load_model):
        print("Could not find existing model, please train model first")
        sys.exit()
    test_batch(model, device, spec_type=get_spec_type())
elif(args.run_training_sweep):
    run_training_sweep()
else:
    print("Loading Dataset")
    dataloader_train, dataloader_val, dataloader_test = build_dataloaders(spec_type=get_spec_type())

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)

    model = model.to(device)
    results = train_model(model, dataloader_train, dataloader_val, optimizer, criterion, device, int(args.epochs))

    # test training
    avg_loss, accuracy = test_model(model, dataloader_test, criterion, device)
    print(f"Average Loss: {avg_loss} | Accuracy: {accuracy}")

    plot_train_results(results, get_spec_type(), show=True)