import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from evaluate import evaluate_model, SignModel

def train_model(train_loader, test_loader, model_path, scaler_path, log_path):
    model = SignModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = StandardScaler()

    epochs = 1000
    train_losses = []
    test_accuracies = []

    with open(log_path, 'w') as log_file:
        log_file.write('Epoch,Loss,Accuracy\n')

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0

            for inputs, labels in train_loader:
                inputs = scaler.fit_transform(inputs.numpy())
                inputs = torch.tensor(inputs, dtype=torch.float32)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            train_losses.append(running_loss / len(train_loader))

            # Evaluate on test set
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs = scaler.transform(inputs.numpy())
                    inputs = torch.tensor(inputs, dtype=torch.float32)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = correct / total
            test_accuracies.append(accuracy)
            log_file.write(f"{epoch},{running_loss / len(train_loader)},{accuracy}\n")
            if epoch % 10 == 0:

                print(f"Epoch {epoch}, Loss: {running_loss / len(train_loader)}, Accuracy: {accuracy}")

    torch.save(model.state_dict(), model_path)
    torch.save(scaler, scaler_path)

if __name__ == "__main__":
    prepared_data_dir = './data/prepared_data'
    log_dir = './data/logs'
    os.makedirs(log_dir, exist_ok=True)

    def load_data(language):
        X_train = np.load(os.path.join(prepared_data_dir, f'X_{language}_train.npy'))
        y_train = np.load(os.path.join(prepared_data_dir, f'y_{language}_train.npy'))
        X_test = np.load(os.path.join(prepared_data_dir, f'X_{language}_test.npy'))
        y_test = np.load(os.path.join(prepared_data_dir, f'y_{language}_test.npy'))

        y_train = torch.tensor(np.argmax(y_train, axis=1), dtype=torch.long)
        y_test = torch.tensor(np.argmax(y_test, axis=1), dtype=torch.long)

        return X_train, y_train, X_test, y_test

    X_asl_train, y_asl_train, X_asl_test, y_asl_test = load_data('asl')
    X_csl_train, y_csl_train, X_csl_test, y_csl_test = load_data('csl')

    train_dataset_asl = TensorDataset(torch.tensor(X_asl_train, dtype=torch.float32), y_asl_train)
    test_dataset_asl = TensorDataset(torch.tensor(X_asl_test, dtype=torch.float32), y_asl_test)
    train_loader_asl = DataLoader(train_dataset_asl, batch_size=32, shuffle=True)
    test_loader_asl = DataLoader(test_dataset_asl, batch_size=32, shuffle=False)

    train_dataset_csl = TensorDataset(torch.tensor(X_csl_train, dtype=torch.float32), y_csl_train)
    test_dataset_csl = TensorDataset(torch.tensor(X_csl_test, dtype=torch.float32), y_csl_test)
    train_loader_csl = DataLoader(train_dataset_csl, batch_size=32, shuffle=True)
    test_loader_csl = DataLoader(test_dataset_csl, batch_size=32, shuffle=False)

    print("Training ASL model...")
    train_model(train_loader_asl, test_loader_asl, 'models/asl_model.pth', 'models/scaler_asl.pth', os.path.join(log_dir, 'asl_training_log.csv'))

    print("Training CSL model...")
    train_model(train_loader_csl, test_loader_csl, 'models/csl_model.pth', 'models/scaler_csl.pth', os.path.join(log_dir, 'csl_training_log.csv'))
