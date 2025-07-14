import pandas as pd
import numpy as np
import os

COLUMNS = [
    'datetime', 'global_active_power', 'global_reactive_power', 'voltage', 'global_intensity',
    'sub_metering_1', 'sub_metering_2', 'sub_metering_3', 'RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU'
]

def process_file(input_path, output_path):
    # 读取数据，强制列名，防止错位
    df = pd.read_csv(input_path, names=COLUMNS, header=0, low_memory=False)
    # 强制数值列为float
    for col in COLUMNS[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # 缺失值处理
    df.ffill(inplace=True)
    df.fillna(0, inplace=True)
    # 解析日期
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = df['datetime'].dt.date
    # 按天聚合
    daily = pd.DataFrame()
    for col in ['global_active_power', 'global_reactive_power', 'sub_metering_1', 'sub_metering_2', 'sub_metering_3']:
        daily[col] = df.groupby('date')[col].sum()
    for col in ['voltage', 'global_intensity']:
        daily[col] = df.groupby('date')[col].mean()
    for col in ['RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU']:
        if col in df.columns:
            daily[col] = df.groupby('date')[col].first()
    daily['sub_metering_remainder'] = (daily['global_active_power'] * 1000 / 60) - (
        daily['sub_metering_1'] + daily['sub_metering_2'] + daily['sub_metering_3'])
    daily.reset_index(inplace=True)
    daily.to_csv(output_path, index=False)

if __name__ == '__main__':
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    process_file(os.path.join(data_dir, 'train.csv'), os.path.join(data_dir, 'train_processed.csv'))
    process_file(os.path.join(data_dir, 'test.csv'), os.path.join(data_dir, 'test_processed.csv'))
    print("Data processed successfully")