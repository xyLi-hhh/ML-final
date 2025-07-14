import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import random
import os

# 固定随机种子
SEED = 42
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed()

# 参数
INPUT_DAYS = 90
OUTPUT_DAYS_LIST = [90, 365]
BATCH_SIZE = 16
EPOCHS = 200
LR = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 读取数据
train_df = pd.read_csv('data/train_processed.csv')
test_df = pd.read_csv('data/test_processed.csv')

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

# 创建middle_data目录
middle_dir = 'middle_data'
os.makedirs(middle_dir, exist_ok=True)

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

class LearnableFreqMask(nn.Module):
    def __init__(self, seq_len, feature_dim):
        super().__init__()
        self.mask_param = nn.Parameter(torch.full((1, seq_len, 1), 0.5))
    def forward(self, freq_x):
        mask = torch.sigmoid(self.mask_param)  # (1, T, 1)
        high_mask = mask
        low_mask = 1 - mask
        high_freq = freq_x * high_mask
        low_freq = freq_x * low_mask
        return high_freq, low_freq, mask

class SimpleTransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    def forward(self, x):
        x = self.input_proj(x)
        x = self.encoder(x)
        return x

class DFTFusionTransformer(nn.Module):
    def __init__(self, seq_len, feature_dim, output_days):
        super().__init__()
        self.freq_mask = LearnableFreqMask(seq_len, feature_dim)
        self.highfreq_trans = SimpleTransformerEncoder(feature_dim)
        self.lowfreq_trans = SimpleTransformerEncoder(feature_dim)
        self.fusion = nn.Linear(128 + 128 + feature_dim, 128)  # 修正此处
        self.decoder = nn.Linear(128, output_days)
    def forward(self, x):  # x: (B, T, C)
        # 1. 频域变换
        x_freq = torch.fft.fft(x, dim=1)
        # 2. 自适应mask分割
        high_freq, low_freq, mask = self.freq_mask(x_freq)
        # 3. 逆DFT回时域
        high_time = torch.fft.ifft(high_freq, dim=1).real
        low_time = torch.fft.ifft(low_freq, dim=1).real
        # 4. 分支Transformer
        high_feat = self.highfreq_trans(high_time)
        low_feat = self.lowfreq_trans(low_time)
        fusion_feat = torch.cat([high_feat[:, -1, :], low_feat[:, -1, :], x[:, -1, :]], dim=-1)
        fusion_feat = self.fusion(fusion_feat)
        # 6. 解码
        out = self.decoder(fusion_feat)
        return out, mask

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
        model = DFTFusionTransformer(INPUT_DAYS, len(FEATURES), OUTPUT_DAYS).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.MSELoss()
        model.train()
        loss_list = []
        for epoch in range(EPOCHS):
            epoch_loss = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred, _ = model(xb)
                loss = criterion(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= len(train_loader.dataset)
            loss_list.append(epoch_loss)
            if (epoch+1) % 10 == 0 or epoch == 0:
                print(f'[DFT] Round {round+1} | {OUTPUT_DAYS} days | Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss:.4f}')
        all_loss_list.append(loss_list)
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
            pred_scaled, mask = model(X_test_tensor)
            pred_scaled = pred_scaled.cpu().numpy().reshape(-1)
        y_true = test_df[TARGET].values[-OUTPUT_DAYS:]
        pred = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(-1)
        y_true = y_true.reshape(-1)
        if round == 0:
            np.save(os.path.join(middle_dir, f'DFT_pred_curve_{OUTPUT_DAYS}.npy'), pred)
            np.save(os.path.join(middle_dir, f'DFT_true_curve_{OUTPUT_DAYS}.npy'), y_true)
            np.save(os.path.join(middle_dir, f'DFT_mask_{OUTPUT_DAYS}.npy'), mask.cpu().numpy())
        mse = mean_squared_error(y_true, pred)
        mae = mean_absolute_error(y_true, pred)
        mse_list.append(mse)
        mae_list.append(mae)
        print(f'[DFT] Round {round+1} | {OUTPUT_DAYS} days prediction: MSE={mse:.2f}, MAE={mae:.2f}')
    mse_mean, mse_std = np.mean(mse_list), np.std(mse_list)
    mae_mean, mae_std = np.mean(mae_list), np.std(mae_list)
    results[OUTPUT_DAYS] = {
        'MSE_mean': mse_mean, 'MSE_std': mse_std,
        'MAE_mean': mae_mean, 'MAE_std': mae_std
    }
    np.save(os.path.join(middle_dir, f'DFT_loss_list_{OUTPUT_DAYS}.npy'), np.array(all_loss_list))
    print(f'\n[DFT] {OUTPUT_DAYS} days prediction result:')
    print(f'MSE: {mse_mean:.2f} ± {mse_std:.2f}')
    print(f'MAE: {mae_mean:.2f} ± {mae_std:.2f}\n')
df_result = pd.DataFrame(results).T
print(df_result)
df_result.to_csv(os.path.join(middle_dir, 'DFT_result.csv')) 