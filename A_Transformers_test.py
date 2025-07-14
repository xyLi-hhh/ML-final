import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import math
import os
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import time

# ======================================================================================
# 1. 日志和环境设置 (Logging and Environment Setup)
# ======================================================================================

os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("results/transformer_training.log", mode='w'),
        logging.StreamHandler()
    ]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")


# ======================================================================================
# 2. Transformer 模型从零实现 (Transformer Model from Scratch)
# ======================================================================================

class PositionalEncoding(nn.Module):
    """
    位置编码模块。
    【修正点】修改了此模块以适应 (batch_size, seq_len, d_model) 的输入形状。
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 将pe的形状从 [max_len, d_model] 变为 [1, max_len, d_model] 以便与批次数据相加
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x 的形状: [batch_size, seq_len, d_model]
        """
        # x.size(1) 是序列长度
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            # mask是2D的，但会被广播到4D的scores上
            scores = scores + mask
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        return output, attn_weights

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        x, _ = self.scaled_dot_product_attention(Q, K, V, mask)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(x)
        return output

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        attn_output = self.self_attn(src, src, src, src_mask)
        src = self.norm1(src + self.dropout1(attn_output))
        ff_output = self.feed_forward(src)
        src = self.norm2(src + self.dropout2(ff_output))
        return src

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask, memory_mask):
        attn_output = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = self.norm1(tgt + self.dropout1(attn_output))
        cross_attn_output = self.cross_attn(tgt, memory, memory, memory_mask)
        tgt = self.norm2(tgt + self.dropout2(cross_attn_output))
        ff_output = self.feed_forward(tgt)
        tgt = self.norm3(tgt + self.dropout3(ff_output))
        return tgt

class Transformer(nn.Module):
    """
    【修正点】统一了所有层的输入形状为 (batch_size, seq_len, features)
    并移除了所有的 .permute() 操作。
    """
    def __init__(self, n_features, n_dec_features, d_model, n_heads, num_encoder_layers, num_decoder_layers, d_ff, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Linear(n_features, d_model)
        self.decoder_embedding = nn.Linear(n_dec_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.pos_decoder = PositionalEncoding(d_model, dropout)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_encoder_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_decoder_layers)])
        self.output_layer = nn.Linear(d_model, 1)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        # 编码器
        src_embed = self.encoder_embedding(src)
        memory = self.pos_encoder(src_embed)
        for layer in self.encoder_layers:
            memory = layer(memory, src_mask)
            
        # 解码器
        tgt_embed = self.decoder_embedding(tgt)
        output = self.pos_decoder(tgt_embed)
        for layer in self.decoder_layers:
            output = layer(output, memory, tgt_mask, memory_mask)
            
        # 输出
        output = self.output_layer(output)
        return output

# ======================================================================================
# 3. 数据处理 (Data Handling)
# ======================================================================================

def load_and_preprocess_data(train_path, test_path):
    try:
        train_df = pd.read_csv(train_path, parse_dates=['DateTime'])
        test_df = pd.read_csv(test_path, parse_dates=['DateTime'])
    except FileNotFoundError as e:
        logging.error(f"Data file not found: {e}. Please ensure CSV files are in the correct directory.")
        return None, None, None, None, None

    logging.info(f"Train data shape: {train_df.shape}")
    logging.info(f"Test data shape: {test_df.shape}")
    
    target_col = 'Global_active_power'
    feature_cols = [col for col in train_df.columns if col not in ['DateTime', target_col]]
    
    logging.info(f"Target column: {target_col}")
    logging.info(f"Feature columns: {feature_cols}")

    full_data_cols = [target_col] + feature_cols
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df[full_data_cols])
    test_scaled = scaler.transform(test_df[full_data_cols])
    
    train_scaled_df = pd.DataFrame(train_scaled, columns=full_data_cols)
    test_scaled_df = pd.DataFrame(test_scaled, columns=full_data_cols)
    
    return train_scaled_df, test_scaled_df, scaler, feature_cols, target_col

def create_inout_sequences(data, input_window, output_window, feature_cols, target_col):
    inout_seq = []
    L = len(data)
    for i in range(L - input_window - output_window + 1):
        # 编码器输入特征
        train_seq = data[feature_cols][i:i+input_window].values
        # 目标序列 (用于解码器输入和标签)
        train_label = data[target_col][i:i+input_window+output_window].values
        inout_seq.append((train_seq, train_label))
    return inout_seq

def prepare_dataloaders(train_data, test_data, input_window, output_window, feature_cols, target_col, batch_size):
    train_sequences = create_inout_sequences(train_data, input_window, output_window, feature_cols, target_col)
    
    # 【修正点】 使用np.array()来提高效率并避免UserWarning
    encoder_input_train = torch.FloatTensor(np.array([seq for seq, _ in train_sequences]))
    decoder_input_train = torch.FloatTensor(np.array([label[:-1] for _, label in train_sequences])).unsqueeze(-1)
    labels_train = torch.FloatTensor(np.array([label[1:] for _, label in train_sequences])).unsqueeze(-1)
    
    train_dataset = TensorDataset(encoder_input_train, decoder_input_train, labels_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 准备测试所需数据
    # encoder_input_test = torch.FloatTensor(train_data[feature_cols][-input_window:].values).unsqueeze(0)
    # decoder_input_test_start = torch.FloatTensor(train_data[target_col][-(input_window-1):].values).unsqueeze(0).unsqueeze(-1)
    # test_labels = torch.FloatTensor(test_data[target_col][:output_window].values).unsqueeze(0).unsqueeze(-1)

    # 测试集只取最后一个滑窗，与LSTM.py一致
    encoder_input_test = torch.FloatTensor(test_data[feature_cols][-input_window:].values).unsqueeze(0)
    # 解码器输入为最后input_window天的target，去掉最后output_window天
    decoder_input_test_start = torch.FloatTensor(test_data[target_col][-input_window:-output_window].values if input_window > output_window else [test_data[target_col].values[-output_window-1]]).unsqueeze(0).unsqueeze(-1)
    # 标签为最后output_window天
    test_labels = torch.FloatTensor(test_data[target_col][-output_window:].values).unsqueeze(0).unsqueeze(-1)
    
    return train_loader, encoder_input_test, decoder_input_test_start, test_labels

# 【新功能】绘制合并的Loss曲线
def plot_combined_loss(loss_histories, horizon, n_epochs):
    plt.figure(figsize=(10, 6))
    for i, history in enumerate(loss_histories):
        plt.plot(range(10, n_epochs + 1, 10), history, label=f'Run {i+1}')
    
    plt.title(f'Combined Training Loss Curves - {horizon}-day Horizon')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'results/combined_loss_{horizon}d.png')
    plt.close()

# 【新功能】绘制合并的预测结果图
def plot_combined_predictions(predictions_list, actuals, horizon):
    predictions_array = np.array(predictions_list)
    mean_preds = np.mean(predictions_array, axis=0)
    std_preds = np.std(predictions_array, axis=0)
    
    plt.figure(figsize=(15, 7))
    # 绘制真实值
    plt.plot(actuals, label='Actual Values', linewidth=2)
    # 绘制平均预测值
    plt.plot(mean_preds, label='Average Prediction', linestyle='--')
    # 绘制标准差阴影区域
    plt.fill_between(
        range(len(mean_preds)),
        mean_preds - std_preds,
        mean_preds + std_preds,
        color='red',
        alpha=0.2,
        label='±1 Std. Dev.'
    )
    
    plt.title(f'Average Prediction vs Actual - {horizon}-day Horizon (5 Runs)')
    plt.xlabel('Time (Days)')
    plt.ylabel('Global Active Power (kW)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'results/combined_prediction_{horizon}d.png')
    plt.close()

# ======================================================================================
# 4. 训练和评估函数 (Training and Evaluation Functions)
# ======================================================================================

def train_model(model, train_loader, optimizer, criterion, epochs, input_window, output_window, run_id):
    # 【修改点】函数现在返回loss_history
    model.train()
    loss_history = []
    
    tgt_mask_size = output_window + input_window - 1
    tgt_mask = model.generate_square_subsequent_mask(tgt_mask_size).to(device)

    logging.info(f"Starting training for run: {run_id}")
    for epoch in range(epochs):
        epoch_loss = 0
        for i, (src, tgt_in, tgt_out) in enumerate(train_loader):
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
            optimizer.zero_grad()
            output = model(src, tgt_in, tgt_mask=tgt_mask)
            loss = criterion(output[:, -output_window:, :], tgt_out[:, -output_window:, :])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        if (epoch + 1) % 10 == 0:
            loss_history.append(avg_epoch_loss)
            logging.info(f'Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.6f}')
            
    logging.info("Training finished.")
    return model, loss_history

def evaluate_model(model, encoder_input, decoder_start_seq, test_labels, scaler, input_window, output_window, target_col_index, run_id):
    # 【修改点】函数现在返回逆缩放后的预测和真实值
    model.eval()
    
    with torch.no_grad():
        encoder_input = encoder_input.to(device)
        decoder_input = decoder_start_seq.to(device)
        predictions = []

        for _ in range(output_window):
            tgt_mask_size = decoder_input.size(1)
            tgt_mask = model.generate_square_subsequent_mask(tgt_mask_size).to(device)
            output = model(encoder_input, decoder_input, tgt_mask=tgt_mask)
            last_pred = output[:, -1:, :]
            predictions.append(last_pred.item())
            decoder_input = torch.cat([decoder_input, last_pred], dim=1)
    
    predictions = np.array(predictions).reshape(-1, 1)
    dummy_preds = np.zeros((len(predictions), scaler.n_features_in_))
    dummy_preds[:, target_col_index] = predictions.flatten()
    predictions_inv = scaler.inverse_transform(dummy_preds)[:, target_col_index]

    labels_squeezed = test_labels.squeeze().numpy()
    dummy_labels = np.zeros((len(labels_squeezed), scaler.n_features_in_))
    dummy_labels[:, target_col_index] = labels_squeezed
    labels_inv = scaler.inverse_transform(dummy_labels)[:, target_col_index]

    mse = mean_squared_error(labels_inv, predictions_inv)
    mae = mean_absolute_error(labels_inv, predictions_inv)
    
    logging.info(f'Evaluation for Horizon {output_window}d - Run {run_id}')
    logging.info(f'MSE: {mse:.4f}, MAE: {mae:.4f}')

    results_df = pd.DataFrame({'Actual': labels_inv, 'Predicted': predictions_inv})
    results_df.to_csv(f'results/predictions_run_{run_id}_horizon_{output_window}.csv', index=False)

    # 新增：每次实验绘制对比图
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(labels_inv, label='Actual', linewidth=2)
    plt.plot(predictions_inv, label='Predicted', linestyle='--')
    plt.title(f'Prediction vs Actual - Run {run_id} ({output_window}-day)')
    plt.xlabel('Time (Days)')
    plt.ylabel('Global Active Power (kW)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'results/pred_vs_actual_run_{run_id}_horizon_{output_window}.png')
    plt.close()
    
    return mse, mae, predictions_inv, labels_inv

# ======================================================================================
# 5. 主执行逻辑 (Main Execution Logic)
# ======================================================================================

if __name__ == '__main__':
    INPUT_WINDOW = 90
    OUTPUT_WINDOWS = [90, 365]
    N_RUNS = 5
    EPOCHS = 200
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    
    D_MODEL = 128
    N_HEADS = 4
    NUM_ENCODER_LAYERS = 4
    NUM_DECODER_LAYERS = 4
    D_FF = 512
    DROPOUT = 0

    train_df, test_df, scaler, feature_cols, target_col = load_and_preprocess_data(
        '/home/xyli/ML-final-main/data/processed_train.csv', '/home/xyli/ML-final-main/data/processed_test.csv'
    )
    
    if train_df is not None:
        n_features = len(feature_cols)
        n_dec_features = 1
        target_col_index = [target_col] + feature_cols
        target_col_index = target_col_index.index(target_col)
        
        for output_window in OUTPUT_WINDOWS:
            logging.info(f"\n{'='*30} STARTING EXPERIMENT: {output_window}-DAY HORIZON {'='*30}")
            
            if len(test_df) < output_window:
                logging.warning(f"Test data is shorter ({len(test_df)} days) than the output window ({output_window} days). Skipping this horizon.")
                continue
            
            # 【新功能】初始化列表以存储所有运行的结果
            all_loss_histories = []
            all_predictions_inv = []
            run_metrics = {'mse': [], 'mae': []}
            ground_truth_labels = None # 用于存储真实标签，只需一次
            
            train_loader, encoder_input_test, decoder_input_test_start, test_labels = prepare_dataloaders(
                train_df, test_df, INPUT_WINDOW, output_window, feature_cols, target_col, BATCH_SIZE
            )
            
            for i in range(1, N_RUNS + 1):
                run_id = f"{output_window}d_{i}"
                logging.info(f"\n--- Running Experiment {i}/{N_RUNS} for {output_window}-day horizon ---")

                model = Transformer(
                    n_features=n_features, n_dec_features=n_dec_features, d_model=D_MODEL,
                    n_heads=N_HEADS, num_encoder_layers=NUM_ENCODER_LAYERS,
                    num_decoder_layers=NUM_DECODER_LAYERS, d_ff=D_FF, dropout=DROPOUT
                ).to(device)

                optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
                criterion = nn.MSELoss()

                start_time = time.time()
                # 【修改点】接收 train_model 返回的 loss_history
                trained_model, loss_history = train_model(
                    model, train_loader, optimizer, criterion, EPOCHS, INPUT_WINDOW, output_window, run_id
                )
                all_loss_histories.append(loss_history)
                training_time = time.time() - start_time
                logging.info(f"Training for run {run_id} completed in {training_time:.2f} seconds.")

                torch.save(trained_model.state_dict(), f'models/transformer_run_{run_id}.pth')

                # 【修改点】接收 evaluate_model 返回的预测和真实值
                mse, mae, predictions_inv, labels_inv = evaluate_model(
                    trained_model, encoder_input_test, decoder_input_test_start, test_labels, scaler, 
                    INPUT_WINDOW, output_window, target_col_index, run_id
                )
                
                run_metrics['mse'].append(mse)
                run_metrics['mae'].append(mae)
                all_predictions_inv.append(predictions_inv)
                if ground_truth_labels is None:
                    ground_truth_labels = labels_inv
            
            # 【新功能】在5次运行结束后，生成合并图表
            logging.info(f"\n--- Generating combined plots for {output_window}-day horizon ---")
            plot_combined_loss(all_loss_histories, output_window, EPOCHS)
            plot_combined_predictions(all_predictions_inv, ground_truth_labels, output_window)
            logging.info(f"Combined plots saved to 'results/' directory.")

            # 汇总并报告5次运行的指标
            mean_mse = np.mean(run_metrics['mse'])
            std_mse = np.std(run_metrics['mse'])
            mean_mae = np.mean(run_metrics['mae'])
            std_mae = np.std(run_metrics['mae'])
            
            logging.info(f"\n{'='*25} FINAL RESULTS for {output_window}-DAY HORIZON (over {N_RUNS} runs) {'='*25}")
            logging.info(f"Average MSE: {mean_mse:.4f} (std: {std_mse:.4f})")
            logging.info(f"Average MAE: {mean_mae:.4f} (std: {std_mae:.4f})")
            logging.info(f"{'='*80}\n")
            
    logging.info("All experiments finished.")