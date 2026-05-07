import os
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(features_dir):
    X = []
    y = []
    for digit in range(1, 12):
        file_path = os.path.join(features_dir, f'{digit}.csv')
        if os.path.exists(file_path):
            data = np.loadtxt(file_path, delimiter=',')
            labels = np.full((data.shape[0], 1), digit - 1)  # Adjusting digit to be zero-indexed
            X.append(data)
            y.append(labels)

    X = np.vstack(X)
    y = np.vstack(y).flatten()

    return X, y

def one_hot_encode(y, num_classes=11):
    return np.eye(num_classes)[y]

def prepare_data(features_dir, output_dir):
    asl_features_dir = os.path.join(features_dir, 'asl_features')
    csl_features_dir = os.path.join(features_dir, 'csl_features')

    X_asl, y_asl = load_data(asl_features_dir)
    X_csl, y_csl = load_data(csl_features_dir)

    y_asl = one_hot_encode(y_asl)
    y_csl = one_hot_encode(y_csl)

    X_asl_train, X_asl_test, y_asl_train, y_asl_test = train_test_split(X_asl, y_asl, test_size=0.2, random_state=42, stratify=y_asl.argmax(axis=1))
    X_csl_train, X_csl_test, y_csl_train, y_csl_test = train_test_split(X_csl, y_csl, test_size=0.2, random_state=42, stratify=y_csl.argmax(axis=1))

    np.save(os.path.join(output_dir, 'X_asl_train.npy'), X_asl_train)
    np.save(os.path.join(output_dir, 'X_asl_test.npy'), X_asl_test)
    np.save(os.path.join(output_dir, 'y_asl_train.npy'), y_asl_train)
    np.save(os.path.join(output_dir, 'y_asl_test.npy'), y_asl_test)
    np.save(os.path.join(output_dir, 'X_csl_train.npy'), X_csl_train)
    np.save(os.path.join(output_dir, 'X_csl_test.npy'), X_csl_test)
    np.save(os.path.join(output_dir, 'y_csl_train.npy'), y_csl_train)
    np.save(os.path.join(output_dir, 'y_csl_test.npy'), y_csl_test)

if __name__ == "__main__":
    features_dir = './data/features'
    output_dir = './data/prepared_data'
    prepare_data(features_dir, output_dir)
