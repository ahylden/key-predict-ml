import matplotlib.pyplot as plt
import os

def plot_train_results(train_results, spec_type, epochs=100, save_dir="training_plots", overlay=False, show=False):

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

chroma_data = [(10, 50.662, 53.146), (20, 55.132, 58.775), (30, 54.636, 57.947),
               (40, 54.967, 57.947), (50, 55.629, 58.940), (60, 55.629, 59.272),
               (70, 55.960, 58.775), (80, 56.954, 59.437), (90, 56.623, 59.437),
               (100, 57.616, 60.762)]
cqt_data = [(10, 67.219, 71.358), (20, 66.556, 70.033), (30, 69.868, 73.013),
            (40, 72.020, 74.834), (50, 74.172, 76.987), (60, 74.172, 76.656),
            (70, 73.841, 76.159), (80, 73.510, 75.662), (90, 74.172, 75.993), 
            (100, 73.510, 76.325)]
logmel_data = [(10, 56.291, 62.583), (20, 56.788, 62.583), (30, 61.093, 65.728),
               (40, 64.570, 69.539), (50, 66.060, 69.702), (60, 64.570, 68.212),
               (70, 66.722, 69.371), (80, 68.377, 70.861), (90, 68.377, 71.026),
               (100, 66.060, 69.040)]

data = []
data.append(chroma_data)
data.append(cqt_data)
data.append(logmel_data)

spec_types = ["chroma", "cqt", "logmel"]

for idx, spec_type in enumerate(spec_types):
    results = data[idx]
    plot_train_results(results, spec_type=spec_type, epochs=100, save_dir="training_plots", overlay=True)
plot_train_results(data, spec_type="logmel", epochs=100, save_dir="training_plots", overlay=True) 