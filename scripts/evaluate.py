import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class SignModel(nn.Module):
    def __init__(self):
        super(SignModel, self).__init__()
        self.fc1 = nn.Linear(63, 128)  # 63 is the input size (21 landmarks * 3 coordinates)
        self.fc2 = nn.Linear(128, 11)  # 11 is the number of output classes (digits 1-10 and exception)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def evaluate_model(test_loader, model_path, scaler):
    model = SignModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    correct = 0
    total = 0
    digit_correct = [0] * 11
    digit_total = [0] * 11

    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = scaler.transform(inputs)
            inputs = torch.tensor(inputs, dtype=torch.float32)
            outputs = model(inputs)
            probabilities = nn.functional.softmax(outputs, dim=1)  # Convert logits to probabilities
            _, predicted = torch.max(probabilities, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

            for label, prediction in zip(labels, predicted):
                digit_total[label.item()] += 1
                if label == prediction:
                    digit_correct[label.item()] += 1

    for digit in range(11):
        digit_accuracy = digit_correct[digit] / digit_total[digit] if digit_total[digit] > 0 else 0
        print(f"Digit {digit + 1}: {digit_accuracy * 100:.2f}% Accuracy")

    overall_accuracy = correct / total
    print(f"Overall Accuracy: {overall_accuracy * 100:.2f}%")

    # # Print some sample predictions for verification
    # print("\nSample Predictions:")
    # for i in range(20):  # Adjust the number of examples to print
    #     true_label = test_loader.dataset[i][1].item()
    #     predicted_label = all_predictions[i]
    #     probabilities = all_probabilities[i]
    #     print(f"Example {i + 1}: True: {true_label + 1}, Predicted: {predicted_label + 1}, Probabilities: {probabilities}")

def plot_training_log(log_path, graph_path):
    epochs, losses, accuracies = [], [], []

    with open(log_path, 'r') as log_file:
        log_file.readline()  # Skip header
        for line in log_file:
            epoch, loss, accuracy = line.strip().split(',')
            epochs.append(int(epoch))
            losses.append(float(loss))
            accuracies.append(float(accuracy))

    plt.figure()
    plt.plot(epochs, losses, label='Loss', color='blue')
    plt.plot(epochs, accuracies, label='Accuracy', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.title('Training Loss and Accuracy')
    plt.savefig(graph_path)
    plt.close()

if __name__ == "__main__":
    prepared_data_dir = './data/prepared_data'
    graph_dir = './data/graphs'
    log_dir = './data/logs'

    def load_data(language):
        X_test = np.load(os.path.join(prepared_data_dir, f'X_{language}_test.npy'))
        y_test = np.load(os.path.join(prepared_data_dir, f'y_{language}_test.npy'))
        return X_test, y_test

    X_asl_test, y_asl_test = load_data('asl')
    X_csl_test, y_csl_test = load_data('csl')

    scaler_asl = torch.load('models/scaler_asl.pth')
    scaler_csl = torch.load('models/scaler_csl.pth')

    test_dataset_asl = TensorDataset(torch.tensor(X_asl_test, dtype=torch.float32), torch.tensor(np.argmax(y_asl_test, axis=1), dtype=torch.long))
    test_loader_asl = DataLoader(test_dataset_asl, batch_size=32, shuffle=False)

    test_dataset_csl = TensorDataset(torch.tensor(X_csl_test, dtype=torch.float32), torch.tensor(np.argmax(y_csl_test, axis=1), dtype=torch.long))
    test_loader_csl = DataLoader(test_dataset_csl, batch_size=32, shuffle=False)

    print("ASL Model Evaluation")
    evaluate_model(test_loader_asl, 'models/asl_model.pth', scaler_asl)

    print("CSL Model Evaluation")
    evaluate_model(test_loader_csl, 'models/csl_model.pth', scaler_csl)

    print("Generating training graphs...")
    plot_training_log(os.path.join(log_dir, 'asl_training_log.csv'), os.path.join(graph_dir, 'asl_training.png'))
    plot_training_log(os.path.join(log_dir, 'csl_training_log.csv'), os.path.join(graph_dir, 'csl_training.png'))
