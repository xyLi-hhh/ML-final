import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob


# matplotlib设置中文显示
plt.rcParams['font.family'] = ['SimHei']

# 1. 自动查找所有 middle_data/*_result.csv，画MSE/MAE均值和标准差
result_files = glob.glob('middle_data/*_result.csv')
for result_path in result_files:
    model_name = os.path.basename(result_path).split('_result.csv')[0]
    df = pd.read_csv(result_path, index_col=0)
    plt.figure(figsize=(10, 5))
    x = np.arange(len(df.index))
    width = 0.35
    plt.bar(x-width/2, df['MSE_mean'], width, yerr=df['MSE_std'], capsize=8, label='MSE', alpha=0.7)
    plt.bar(x+width/2, df['MAE_mean'], width, yerr=df['MAE_std'], capsize=8, label='MAE', alpha=0.7)
    plt.ylabel('Score')
    plt.xlabel('Prediction Days')
    plt.title(f'{model_name} Model MSE/MAE Mean and Std')
    plt.xticks(x, df.index.astype(str))
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{model_name}_MSE_MAE_bar.png')
    plt.show()

# 2. 自动查找所有 middle_data/*_loss_list_*.npy，画loss曲线
loss_files = glob.glob('middle_data/*_loss_list_*.npy')
for loss_path in loss_files:
    base = os.path.basename(loss_path)
    model_name = base.split('_loss_list_')[0]
    days = base.split('_loss_list_')[1].replace('.npy','')
    loss_list = np.load(loss_path)
    plt.figure(figsize=(10, 5))
    for i, loss_curve in enumerate(loss_list):
        plt.plot(loss_curve, label=f'Round {i+1}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} {days}-Day Prediction Loss Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{model_name}_loss_curve_{days}.png')
    plt.show()

# 3. 自动查找所有 middle_data/*_pred_curve_*.npy 和 *_true_curve_*.npy，画预测曲线和真实曲线
pred_files = glob.glob('middle_data/*_pred_curve_*.npy')
for pred_path in pred_files:
    base = os.path.basename(pred_path)
    model_name = base.split('_pred_curve_')[0]
    days = base.split('_pred_curve_')[1].replace('.npy','')
    true_path = os.path.join('middle_data', f'{model_name}_true_curve_{days}.npy')
    if os.path.exists(true_path):
        pred = np.load(pred_path)
        true = np.load(true_path)
        plt.figure(figsize=(12, 5))
        plt.plot(true, label='True')
        plt.plot(pred, label=f'{model_name} Prediction')
        plt.xlabel('Days')
        plt.ylabel('global_active_power')
        plt.title(f'{model_name} {days}-Day Prediction vs True')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{model_name}_curve_{days}.png')
        plt.show()
    else:
        print(f'Not found: {true_path}, please make sure true curve is saved.')

# 4. 电力消耗曲线的规律性说明
print("\n【电力消耗曲线规律性说明】")
print("1. 家庭总有功功率（global_active_power）通常呈现周期性波动，受季节、周末/工作日、节假日等影响。")
print("2. 曲线应有一定的平稳性，短期内不会剧烈跳变，长期趋势可能随季节变化缓慢上升或下降。")
print("3. 如果预测曲线与真实曲线趋势一致，且误差较小，说明模型捕捉到了主要规律。")
print("4. 若出现异常尖峰或剧烈波动，需检查数据或模型是否存在问题。") 