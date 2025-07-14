import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import math
import matplotlib.pyplot as plt
import os
import logging
import random

# -------------------------------------------
# 1. 手动实现 LSTM 单元 (LSTMCell)
# -------------------------------------------
class LSTMCell(nn.Module):
    """
    一个手动实现的LSTM单元。
    这个单元处理单个时间步的计算，对应着LSTM的核心公式。
    """
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_ih = nn.Parameter(torch.Tensor(input_size, hidden_size * 4))
        self.W_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size * 4))
        self.b_ih = nn.Parameter(torch.Tensor(hidden_size * 4))
        self.b_hh = nn.Parameter(torch.Tensor(hidden_size * 4))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, states):
        h_prev, c_prev = states
        gates = (torch.matmul(x, self.W_ih) + self.b_ih) + (torch.matmul(h_prev, self.W_hh) + self.b_hh)
        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, 1)
        f_t = torch.sigmoid(forget_gate)
        i_t = torch.sigmoid(input_gate)
        o_t = torch.sigmoid(output_gate)
        g_t = torch.tanh(cell_gate)
        c_next = f_t * c_prev + i_t * g_t
        h_next = o_t * torch.tanh(c_next)
        return h_next, c_next

# -------------------------------------------
# 2. 构建完整的 LSTM 模型
# -------------------------------------------
class MyLSTM(nn.Module):
    """
    一个完整的、基于我们手动实现的LSTMCell的LSTM模型。
    这个模型可以处理一个序列的输入，并输出一个预测值。
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(MyLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm_cells = nn.ModuleList()
        self.lstm_cells.append(LSTMCell(input_size, hidden_size))
        for _ in range(num_layers - 1):
            self.lstm_cells.append(LSTMCell(hidden_size, hidden_size))
        self.fc = nn.Linear(hidden_size, output_size)

    def _init_states(self, batch_size, device):
        return [(torch.zeros(batch_size, self.hidden_size, device=device),
                 torch.zeros(batch_size, self.hidden_size, device=device)) for _ in range(self.num_layers)]

    def forward(self, x, states=None):
        batch_size, seq_len, _ = x.shape
        if states is None:
            h_c_states = self._init_states(batch_size, x.device)
        else:
            h_c_states = states
        for t in range(seq_len):
            input_t = x[:, t, :]
            for layer_idx, lstm_cell in enumerate(self.lstm_cells):
                h_prev, c_prev = h_c_states[layer_idx]
                h_next, c_next = lstm_cell(input_t, (h_prev, c_prev))
                h_c_states[layer_idx] = (h_next, c_next)
                input_t = h_next
        last_layer_h = h_c_states[-1][0]
        output = self.fc(last_layer_h)
        return output

# -------------------------------------------
# 3. 数据处理和训练流程
# -------------------------------------------
def create_multivariate_sequences(input_data, output_data, input_window, output_window):
    """
    为多变量时间序列数据创建输入-输出对。
    """
    X, y = [], []
    for i in range(len(input_data) - input_window - output_window + 1):
        X.append(input_data[i:(i + input_window), :])
        y.append(output_data[(i + input_window):(i + input_window + output_window)])
    return np.array(X), np.array(y)

# -------------------------------------------
# 4. 单轮训练 + 预测函数 (修改: 返回loss列表和预测值)
# -------------------------------------------
def train_and_evaluate(run_idx, is_short=True, params=None, train_df=None, test_df=None, device=None, scaler_X=None, scaler_y=None, input_features=None):
    """
    单轮训练和评估函数。
    Args:
        ... (同前)
    Returns:
        tuple: (mse, mae, train_loss_list, pred_values)
               mse, mae: 本轮的MSE和MAE。
               train_loss_list: 本轮训练过程中的loss列表。
               pred_values: 本轮对测试集的预测结果数组。
    """
    term_name = "Short" if is_short else "Long"
    logging.info(f"========== Starting {term_name}-term Prediction: Run {run_idx+1} ==========")

    input_window = params['INPUT_WINDOW']
    output_window = params['OUTPUT_WINDOW']
    hidden_size = params['HIDDEN_SIZE']
    num_layers = params['NUM_LAYERS']
    epochs = params['EPOCHS']
    batch_size = params['BATCH_SIZE']
    lr = params['LR']

    train_target = train_df['global_active_power'].values.reshape(-1, 1)
    train_features = train_df[input_features].values
    train_features_norm = scaler_X.transform(train_features)
    train_target_norm = scaler_y.transform(train_target)

    X_train, y_train = create_multivariate_sequences(train_features_norm, train_target_norm, input_window, output_window)
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).float().squeeze(-1)
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    INPUT_SIZE = train_features.shape[1]
    model = MyLSTM(INPUT_SIZE, hidden_size, output_window, num_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loss_list = []

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(loader)
        train_loss_list.append(avg_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == epochs:
            logging.info(f"[Run {run_idx+1}] Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")

    # 保存单次运行的训练曲线
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), train_loss_list, marker=' ')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{term_name}-term Training Loss Curve (Run {run_idx+1})')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'individual_loss_{"short" if is_short else "long"}_run{run_idx+1}.png')
    plt.close()

    # # 测试集滚动预测
    # history = pd.concat([train_df[input_features], test_df[input_features]], ignore_index=True)
    # history_norm = scaler_X.transform(history.values.astype(float))
    # true_values = test_df['global_active_power'].values

    # all_predictions = []
    # model.eval()
    # with torch.no_grad():
    #     for i in range(0, len(test_df), output_window):
    #         start_idx = len(train_df) + i - input_window
    #         end_idx = len(train_df) + i
    #         input_sequence = history_norm[start_idx:end_idx]
    #         input_tensor = torch.from_numpy(input_sequence).unsqueeze(0).float().to(device)
    #         pred_norm = model(input_tensor).squeeze(0).cpu().numpy()
    #         all_predictions.append(pred_norm)

    # full_pred_norm = np.concatenate(all_predictions, axis=0)[:len(test_df)]
    # pred_values = scaler_y.inverse_transform(full_pred_norm.reshape(-1, 1)).flatten()

    # 只用测试集最后一个滑窗进行预测
    test_features_norm = scaler_X.transform(test_df[input_features].values.astype(float))
    input_sequence = test_features_norm[-input_window:]  # 取test最后input_window天
    input_tensor = torch.from_numpy(input_sequence).unsqueeze(0).float().to(device)
    model.eval()
    with torch.no_grad():
        pred_norm = model(input_tensor).squeeze(0).cpu().numpy()

    # 只预测output_window天
    pred_values = scaler_y.inverse_transform(pred_norm.reshape(-1, 1)).flatten()
    true_values = test_df['global_active_power'].values[-output_window:]

    # 保存单次运行的结果
    results_df = pd.DataFrame({
        'DateTime': test_df['date'].values[-output_window:],
        'True_Global_active_power': true_values,
        'Predicted_Global_active_power': pred_values
    })
    output_file = f'individual_results_{"short" if is_short else "long"}_run{run_idx+1}.csv'
    results_df.to_csv(output_file, index=False)
    logging.info(f"Individual prediction results for Run {run_idx+1} saved to {output_file}")

    # 计算指标
    mse = np.mean((pred_values - true_values) ** 2)
    mae = np.mean(np.abs(pred_values - true_values))
    logging.info(f"Run {run_idx+1} - MSE: {mse:.4f}, MAE: {mae:.4f}")

    # 绘制单次运行的对比图
    plt.figure(figsize=(15, 6))
    n_plot = output_window if is_short else min(len(test_df), 365) # 长期只画前365天，避免图太乱
    plt.plot(results_df['DateTime'][:n_plot], results_df['True_Global_active_power'][:n_plot], label='True')
    plt.plot(results_df['DateTime'][:n_plot], results_df['Predicted_Global_active_power'][:n_plot], linestyle='--', label='Predicted')
    plt.title(f'{term_name}-term: True vs Predicted (Run {run_idx+1})')
    plt.xlabel('Date')
    plt.ylabel('Global Active Power (kW)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'individual_prediction_{"short" if is_short else "long"}_run{run_idx+1}.png')
    plt.close()

    logging.info(f"========== Finished {term_name}-term Prediction: Run {run_idx+1} ==========\n")
    # 返回评估指标、loss列表和预测值
    return mse, mae, train_loss_list, pred_values

# -------------------------------------------
# 5. 主函数：多轮评估与汇总
# -------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    
    # ====================== 日志配置 ======================
    log_file = 'training_run_summary.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging configured. Output will be saved to training_run_summary.log")

    # ====================== 配置 ======================
    NUM_RUNS = 5

    SHORT_PARAMS = {
        'INPUT_WINDOW': 90, 'OUTPUT_WINDOW': 90, 'HIDDEN_SIZE': 128,
        'NUM_LAYERS': 3, 'EPOCHS': 200, 'BATCH_SIZE': 32, 'LR': 1e-4,
    }
    LONG_PARAMS = {
        'INPUT_WINDOW': 90, 'OUTPUT_WINDOW': 365, 'HIDDEN_SIZE': 128,
        'NUM_LAYERS': 3, 'EPOCHS': 200, 'BATCH_SIZE': 32, 'LR': 1e-4,
    }

    # !! 请确保路径正确 !!
    train_path = '/home/xyli/ML-final-main/ML-3/data/train_processed.csv'
    test_path = '/home/xyli/ML-final-main/ML-3/data/test_processed.csv'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        logging.error(f"Data file not found. Checked paths: {train_path}, {test_path}")
        return

    train_df = pd.read_csv(train_path, parse_dates=['date'])
    test_df = pd.read_csv(test_path, parse_dates=['date'])

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    input_features = [col for col in train_df.columns if col not in ['date']]
    scaler_X.fit(train_df[input_features].values.astype(float))
    scaler_y.fit(train_df['global_active_power'].values.reshape(-1, 1))

    # 初始化用于存储所有运行结果的列表
    all_short_metrics, all_long_metrics = [], []
    all_short_losses, all_long_losses = [], []
    all_short_predictions, all_long_predictions = [], []

    # ------------------ 短期预测 ------------------
    for run in range(NUM_RUNS):
        set_seed(42+run)
        mse, mae, loss_list, predictions = train_and_evaluate(
            run, is_short=True, params=SHORT_PARAMS,
            train_df=train_df, test_df=test_df, device=device,
            scaler_X=scaler_X, scaler_y=scaler_y, input_features=input_features)
        all_short_metrics.append({'mse': mse, 'mae': mae})
        all_short_losses.append(loss_list)
        all_short_predictions.append(predictions)

    # ------------------ 长期预测 ------------------
    for run in range(NUM_RUNS):
        set_seed(42+run)
        mse, mae, loss_list, predictions = train_and_evaluate(
            run, is_short=False, params=LONG_PARAMS,
            train_df=train_df, test_df=test_df, device=device,
            scaler_X=scaler_X, scaler_y=scaler_y, input_features=input_features)
        all_long_metrics.append({'mse': mse, 'mae': mae})
        all_long_losses.append(loss_list)
        all_long_predictions.append(predictions)

    # ====================== 汇总与可视化 ======================
    logging.info("\n" + "="*20 + " FINAL SUMMARY " + "="*20)

    # 1. 汇总指标计算与打印
    short_mse_list = [m['mse'] for m in all_short_metrics]
    short_mae_list = [m['mae'] for m in all_short_metrics]
    long_mse_list = [m['mse'] for m in all_long_metrics]
    long_mae_list = [m['mae'] for m in all_long_metrics]

    summary_df = pd.DataFrame({
        'Metric': ['MSE (Mean)', 'MSE (Std)', 'MAE (Mean)', 'MAE (Std)'],
        'Short-term': [np.mean(short_mse_list), np.std(short_mse_list),
                       np.mean(short_mae_list), np.std(short_mae_list)],
        'Long-term': [np.mean(long_mse_list), np.std(long_mse_list),
                      np.mean(long_mae_list), np.std(long_mae_list)]
    })
    logging.info(f"\nPerformance Summary over {NUM_RUNS} runs:\n{summary_df.to_string(index=False)}")

    # 2. 绘制汇总的Loss曲线
    def plot_combined_loss(all_losses, term_name, epochs):
        plt.figure(figsize=(10, 6))
        for i, loss_list in enumerate(all_losses):
            plt.plot(range(1, epochs + 1), loss_list, label=f'Run {i+1}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Combined {term_name}-term Training Loss ({NUM_RUNS} Runs)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        filename = f'summary_loss_combined_{term_name.lower()}.png'
        plt.savefig(filename)
        plt.close()
        logging.info(f"Combined loss plot saved to {filename}")

    plot_combined_loss(all_short_losses, "Short", SHORT_PARAMS['EPOCHS'])
    plot_combined_loss(all_long_losses, "Long", LONG_PARAMS['EPOCHS'])
    
    # 3. 绘制汇总的预测对比图（平均值+方差）
    true_values = test_df['global_active_power'].values
    timestamps = test_df['date']
    def plot_combined_predictions(all_preds, true_vals, times, term_name, n_plot_days):
        plt.figure(figsize=(15, 7))
        all_preds_np = np.array(all_preds)
        mean_preds = np.mean(all_preds_np, axis=0)
        
        # 截取要绘制的部分
        true_slice = true_vals[:n_plot_days]
        mean_slice = mean_preds[:n_plot_days]
        time_slice = times[:n_plot_days]

        # 绘制所有单次运行的预测（半透明）
        for i in range(len(all_preds)):
            pred_slice = all_preds[i][:n_plot_days]
            plt.plot(time_slice, pred_slice, color='gray', alpha=0.3, linestyle='--')
        
        # 绘制真实值和平均预测值
        plt.plot(time_slice, true_slice, label='True Values', color='blue', linewidth=2)
        plt.plot(time_slice, mean_slice, label=f'Average Prediction ({NUM_RUNS} runs)', color='red', linestyle='-', linewidth=2.5)

        plt.title(f'Combined {term_name}-term Prediction vs. True Values')
        plt.xlabel('Date')
        plt.ylabel('Global Active Power (kW)')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        filename = f'summary_prediction_combined_{term_name.lower()}.png'
        plt.savefig(filename)
        plt.close()
        logging.info(f"Combined prediction plot saved to {filename}")

    plot_combined_predictions(all_short_predictions, true_values, timestamps, "Short", SHORT_PARAMS['OUTPUT_WINDOW'])
    plot_combined_predictions(all_long_predictions, true_values, timestamps, "Long", min(len(test_df), 365))

    # 4. 绘制最终性能对比条形图
    labels = ['Short-term', 'Long-term']
    mse_means = [np.mean(short_mse_list), np.mean(long_mse_list)]
    mae_means = [np.mean(short_mae_list), np.mean(long_mae_list)]
    mse_std = [np.std(short_mse_list), np.std(long_mse_list)]
    mae_std = [np.std(short_mae_list), np.std(long_mae_list)]

    x = np.arange(len(labels))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # MSE Bar Chart
    ax1.bar(x, mse_means, width, yerr=mse_std, capsize=5, label='Mean MSE')
    ax1.set_ylabel('Scores')
    ax1.set_title(f'Mean Squared Error (MSE) over {NUM_RUNS} Runs')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()

    # MAE Bar Chart
    ax2.bar(x, mae_means, width, yerr=mae_std, capsize=5, label='Mean MAE', color='orange')
    ax2.set_ylabel('Scores')
    ax2.set_title(f'Mean Absolute Error (MAE) over {NUM_RUNS} Runs')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend()
    
    fig.tight_layout()
    filename = 'summary_metrics_comparison.png'
    plt.savefig(filename)
    plt.close()
    logging.info(f"Metrics comparison bar chart saved to {filename}")
    
    logging.info("\n" + "="*53)
    logging.info("All tasks completed. Check the generated .png and .log files for results.")


if __name__ == '__main__':
    main()