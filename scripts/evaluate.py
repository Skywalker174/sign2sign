import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import joblib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SignModel(nn.Module):
    def __init__(self):
        super(SignModel, self).__init__()
        self.fc1 = nn.Linear(63, 128)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 11)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def evaluate_model(test_loader, model_path, scaler):
    model = SignModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    correct = 0
    total = 0
    digit_correct = [0] * 11
    digit_total = [0] * 11

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs_scaled = scaler.transform(inputs.numpy())
            inputs_tensor = torch.tensor(inputs_scaled, dtype=torch.float32).to(device)
            labels = labels.to(device)
            
            outputs = model(inputs_tensor)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for label, prediction in zip(labels, predicted):
                digit_total[label.item()] += 1
                if label == prediction:
                    digit_correct[label.item()] += 1

    print("\nDetailed Accuracy per Digit:")
    for digit in range(11):
        if digit_total[digit] > 0:
            digit_accuracy = digit_correct[digit] / digit_total[digit]
            print(f"Digit {digit + 1}: {digit_accuracy * 100:.2f}%")
        else:
            print(f"Digit {digit + 1}: No test data")

    overall_accuracy = correct / total
    print(f"\nOverall Accuracy: {overall_accuracy * 100:.2f}%")

def plot_training_log(log_path, graph_path):
    if not os.path.exists(log_path):
        print(f"Warning: Log file {log_path} not found.")
        return

    epochs, losses, accuracies = [], [], []
    with open(log_path, 'r') as log_file:
        log_file.readline()
        for line in log_file:
            parts = line.strip().split(',')
            if len(parts) == 3:
                epoch, loss, accuracy = parts
                epochs.append(int(epoch))
                losses.append(float(loss))
                accuracies.append(float(accuracy))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, label='Loss', color='blue')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, label='Accuracy', color='red')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig(graph_path)
    plt.close()

if __name__ == "__main__":
    prepared_data_dir = './data/prepared_data'
    graph_dir = './data/graphs'
    log_dir = './data/logs'
    os.makedirs(graph_dir, exist_ok=True)

    def load_data(language):
        X_test = np.load(os.path.join(prepared_data_dir, f'X_{language}_test.npy'))
        y_test = np.load(os.path.join(prepared_data_dir, f'y_{language}_test.npy'))
        return X_test, y_test

    scaler_asl = joblib.load('models/scaler_asl.pkl')
    scaler_csl = joblib.load('models/scaler_csl.pkl')

    print("--- ASL Model Evaluation ---")
    X_asl, y_asl = load_data('asl')
    test_ds_asl = TensorDataset(
        torch.tensor(X_asl, dtype=torch.float32), 
        torch.tensor(np.argmax(y_asl, axis=1), dtype=torch.long)
    )
    loader_asl = DataLoader(test_ds_asl, batch_size=64, shuffle=False)
    evaluate_model(loader_asl, 'models/asl_model.pth', scaler_asl)

    print("\n--- CSL Model Evaluation ---")
    X_csl, y_csl = load_data('csl')
    test_ds_csl = TensorDataset(
        torch.tensor(X_csl, dtype=torch.float32), 
        torch.tensor(np.argmax(y_csl, axis=1), dtype=torch.long)
    )
    loader_csl = DataLoader(test_ds_csl, batch_size=64, shuffle=False)
    evaluate_model(loader_csl, 'models/csl_model.pth', scaler_csl)

    print("\nGenerating training graphs...")
    plot_training_log(os.path.join(log_dir, 'asl_training_log.csv'), os.path.join(graph_dir, 'asl_training.png'))
    plot_training_log(os.path.join(log_dir, 'csl_training_log.csv'), os.path.join(graph_dir, 'csl_training.png'))