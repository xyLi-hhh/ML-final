import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import random
import logging
import os

SEED = 42
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed()

INPUT_DAYS = 90
OUTPUT_DAYS_LIST = [90, 365]
BATCH_SIZE = 16
EPOCHS = 200
LR = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_df = pd.read_csv('/home/xyli/ML-final-main/ML-3/data/train_processed.csv')
test_df = pd.read_csv('/home/xyli/ML-final-main/ML-3/data/test_processed.csv')

FEATURES = [
    'global_active_power', 'global_reactive_power', 'sub_metering_1', 'sub_metering_2', 'sub_metering_3',
    'voltage', 'global_intensity', 'RR', 'NBJRR1', 'NBJRR5', 'NBJBROU', 'sub_metering_remainder'
]
TARGET = 'global_active_power'

scaler = StandardScaler()
train_features = scaler.fit_transform(train_df[FEATURES])
test_features = scaler.transform(test_df[FEATURES])

target_scaler = StandardScaler()
train_target_scaled = target_scaler.fit_transform(train_df[[TARGET]]).reshape(-1)

def make_samples(features, target, input_days, output_days):
    X, y = [], []
    for i in range(len(features) - input_days - output_days + 1):
        X.append(features[i:i+input_days])
        y.append(target[i+input_days:i+input_days+output_days])
    return np.array(X), np.array(y)

class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class TransModel(nn.Module):
    def __init__(self, input_size, output_days, d_model=128, nhead=8, num_layers=3, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_days)
    def forward(self, x):
        x = self.input_proj(x)
        x = self.encoder(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

# ====================== 日志配置 ======================
log_file = 'training_run_summary_trans.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler()
    ]
)
logging.info("Logging configured. Output will be saved to training_run_summary.log")

# 创建middle_data目录
middle_dir = 'middle_data'
os.makedirs(middle_dir, exist_ok=True)

results = {}
for OUTPUT_DAYS in OUTPUT_DAYS_LIST:
    mse_list, mae_list = [], []
    all_loss_list = []
    for round in range(5):
        set_seed(SEED + round)
        X_train, y_train = make_samples(train_features, train_target_scaled, INPUT_DAYS, OUTPUT_DAYS)
        X_test, _ = make_samples(test_features, np.zeros(len(test_features)), INPUT_DAYS, OUTPUT_DAYS)
        X_test = X_test[-1:]
        train_dataset = SeqDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        model = TransModel(len(FEATURES), OUTPUT_DAYS).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.MSELoss()
        model.train()
        loss_list = []
        for epoch in range(EPOCHS):
            epoch_loss = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                loss = criterion(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= len(train_loader.dataset)
            loss_list.append(epoch_loss)
            if (epoch+1) % 10 == 0 or epoch == 0:
                logging.info(f'[Transformer] Round {round+1} | {OUTPUT_DAYS} days | Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss:.4f}')
        all_loss_list.append(loss_list)
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
            pred_scaled = model(X_test_tensor).cpu().numpy().reshape(-1)
        y_true = test_df[TARGET].values[-OUTPUT_DAYS:]
        pred = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(-1)
        y_true = y_true.reshape(-1)
        if round == 0:
            np.save(os.path.join(middle_dir, f'Trans_pred_curve_{OUTPUT_DAYS}.npy'), pred)
            np.save(os.path.join(middle_dir, f'Trans_true_curve_{OUTPUT_DAYS}.npy'), y_true)
        mse = mean_squared_error(y_true, pred)
        mae = mean_absolute_error(y_true, pred)
        mse_list.append(mse)
        mae_list.append(mae)
        logging.info(f'[Transformer] Round {round+1} | {OUTPUT_DAYS} days prediction: MSE={mse:.2f}, MAE={mae:.2f}')
    mse_mean, mse_std = np.mean(mse_list), np.std(mse_list)
    mae_mean, mae_std = np.mean(mae_list), np.std(mae_list)
    results[OUTPUT_DAYS] = {
        'MSE_mean': mse_mean, 'MSE_std': mse_std,
        'MAE_mean': mae_mean, 'MAE_std': mae_std
    }
    np.save(os.path.join(middle_dir, f'Trans_loss_list_{OUTPUT_DAYS}.npy'), np.array(all_loss_list))
    logging.info(f'\n[Transformer] {OUTPUT_DAYS} days prediction result:')
    logging.info(f'MSE: {mse_mean:.2f} ± {mse_std:.2f}')
    logging.info(f'MAE: {mae_mean:.2f} ± {mae_std:.2f}\n')
df_result = pd.DataFrame(results).T
logging.info(df_result)
df_result.to_csv(os.path.join(middle_dir, 'Trans_result.csv')) 