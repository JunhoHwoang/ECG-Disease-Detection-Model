# CNN(4) + BiLSTM(3) + Attention + FocalLoss model
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.signal
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from helper_code import *
from preprocess_ecg import preprocess_ecg

################################################################################
# Focal Loss
################################################################################
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.9, gamma=1.5):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = 1e-9
    def forward(self, input, target):
        log_probs = input
        probs = torch.exp(log_probs)
        target_one_hot = torch.zeros_like(input).scatter(1, target.unsqueeze(1), 1)
        pt = (probs * target_one_hot).sum(1) + self.eps
        log_pt = torch.log(pt)
        focal_term = (1 - pt) ** self.gamma
        loss = -self.alpha * focal_term * log_pt
        return loss.mean()
################################################################################
# Training
################################################################################
def train_model(data_folder, model_folder, verbose):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(data_folder, str):
        data_folders = [data_folder]
    else:
        data_folders = data_folder

    features, labels = preload_data(data_folders, verbose)
    dataset = ChallengeDataset(features, labels)

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    model = CnnLstmAttnModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = FocalLoss(alpha=0.9, gamma=1.5)
    model.train()
    for epoch in tqdm(range(20), desc="Epochs"):
        total_loss = 0
        for features, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False):
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(features)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * features.size(0)
        if verbose:
            print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataset):.4f}")
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save({"state_dict": model.state_dict()}, os.path.join(model_folder, "lstm_model.pth"))
################################################################################
# Load model
################################################################################
def load_model(model_folder, verbose):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(model_folder, "lstm_model.pth")
    model = CnnLstmAttnModel().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model
################################################################################
# Run model
################################################################################
def run_model(record, model, verbose):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = extract_features(record)
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        log_probs = model(features_tensor)
        probs = torch.exp(log_probs)
        prob = probs[0, 1].item()
    return prob > 0.3, prob

################################################################################
# Preloaded Dataset
################################################################################
class ChallengeDataset(Dataset):
    def __init__(self, feature_list, label_list):
        self.features = feature_list
        self.labels = label_list

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

################################################################################
# Preload and preprocess data
################################################################################
def preload_data(data_folders, verbose=False):
    if isinstance(data_folders, str):
        data_folders = [data_folders]

    records = []

    for folder in data_folders:
        for record in find_records(folder):
            path = os.path.join(folder, record)
            header = load_header(path)
            length = get_num_samples(header)
            
            if length < 2800:
                continue

            records.append(path)
            
    features_list = []
    labels_list = []

    if verbose:
        for path in tqdm(records, desc="Preprocessing data"):
            features = extract_features(path)
            label = int(load_label(path))
            features_list.append(torch.tensor(features, dtype=torch.float32))
            labels_list.append(torch.tensor(label, dtype=torch.long))
    else:
        for path in records:
            features = extract_features(path)
            label = int(load_label(path))
            features_list.append(torch.tensor(features, dtype=torch.float32))
            labels_list.append(torch.tensor(label, dtype=torch.long))

    return features_list, labels_list

def extract_features(path):
    signal, fields = load_signals(path)
    features = preprocess_ecg(signal, fields, target_length=2934) 
    features = np.transpose(features, (1, 0))
    return features
################################################################################
# Attention module
################################################################################
class Attention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1)
        )
    def forward(self, lstm_output):
        attn_weights = self.attn(lstm_output)  # (B, T, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        weighted_sum = torch.sum(attn_weights * lstm_output, dim=1)  # (B, H)
        return weighted_sum
################################################################################
# CNN + BiLSTM + Attention model
################################################################################
class CnnLstmAttnModel(nn.Module):
    def __init__(self, input_channels=12, lstm_hidden=128, num_classes=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=lstm_hidden,
                            num_layers=3, batch_first=True,
                            dropout=0.5, bidirectional=True)
        self.attn = Attention(lstm_hidden * 2)
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)
    def forward(self, x):
        x = self.cnn(x)           # (B, C, T)
        x = x.permute(0, 2, 1)    # (B, T, C)
        out, _ = self.lstm(x)     # (B, T, 2H)
        out = self.attn(out)      # (B, 2H)
        out = self.fc(out)        # (B, classes)
        return self.log_softmax(out)