import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib

# Device configuration
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

def train_model(X_train, y_train, X_test, y_test, model_path, scaler_path, log_path):
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert to Tensors and move to device
    train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), y_train)
    test_dataset = TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32), y_test)
    
    # Increased batch size to leverage RTX 5080
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = SignModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 1000
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    with open(log_path, 'w') as log_file:
        log_file.write('Epoch,Loss,Accuracy\n')

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # Evaluation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = correct / total
            avg_loss = running_loss / len(train_loader)
            log_file.write(f"{epoch},{avg_loss},{accuracy}\n")
            
            if epoch % 50 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    # Save model weights and scaler
    torch.save(model.cpu().state_dict(), model_path)
    joblib.dump(scaler, scaler_path) 

if __name__ == "__main__":
    prepared_data_dir = './data/prepared_data'
    log_dir = './data/logs'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs('models', exist_ok=True)

    def load_data(language):
        X_train = np.load(os.path.join(prepared_data_dir, f'X_{language}_train.npy'))
        y_train = np.load(os.path.join(prepared_data_dir, f'y_{language}_train.npy'))
        X_test = np.load(os.path.join(prepared_data_dir, f'X_{language}_test.npy'))
        y_test = np.load(os.path.join(prepared_data_dir, f'y_{language}_test.npy'))

        y_train_idx = torch.tensor(np.argmax(y_train, axis=1), dtype=torch.long)
        y_test_idx = torch.tensor(np.argmax(y_test, axis=1), dtype=torch.long)

        return X_train, y_train_idx, X_test, y_test_idx

    print(f"Training on: {device}")
    
    print("--- Training ASL Model ---")
    X_asl_tr, y_asl_tr, X_asl_te, y_asl_te = load_data('asl')
    train_model(X_asl_tr, y_asl_tr, X_asl_te, y_asl_te, 
                'models/asl_model.pth', 'models/scaler_asl.pkl', 
                os.path.join(log_dir, 'asl_training_log.csv'))

    print("\n--- Training CSL Model ---")
    X_csl_tr, y_csl_tr, X_csl_te, y_csl_te = load_data('csl')
    train_model(X_csl_tr, y_csl_tr, X_csl_te, y_csl_te, 
                'models/csl_model.pth', 'models/scaler_csl.pkl', 
                os.path.join(log_dir, 'csl_training_log.csv'))

    print("\nAll models trained and saved successfully.")