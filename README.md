# 时序预测项目

## 项目简介

本项目基于多种深度学习模型（LSTM、DFT、Transformer）实现多变量时序数据的预测。

## 目录结构

```
ML-final/
├── data/                  # 原始与处理后数据
├── middle_data/           # 中间结果
├── lstm_result/           # LSTM模型输出
├── dft_result/            # DFT模型输出
├── trans_result/          # Transformer模型输出
├── process_data.py        # 数据预处理脚本
├── LSTM.py                # LSTM接口模型定义和训练
├── DFT.py                 # DFT模型定义与训练
├── Trans.py               # Transformer模型定义与训练
├── A_LSTM_test.py         # LSTM自定义实验
├── A_Transformers_test.py # Transformer自定义实验
├── outcome.py             # 结果汇总与可视化
└── README.md              # 项目说明文档
```

## 依赖环境

- Python 3.7+
- numpy
- pandas
- torch
- scikit-learn
- matplotlib

安装依赖：
```bash
pip install -r requirements.txt
```

## 数据准备

请将原始数据放入`data/`目录，并根据需要运行`process_data.py`进行预处理。  
训练和测试数据文件名需与脚本中一致（如`train_processed.csv`、`test_processed.csv`）。

## 运行方法

Transformer：

```bash
python Trans.py
```

LSTM：

```bash
python A_LSTM_test.py
```

DFT:

```bash
python DFT.py
```

## 主要文件说明

- `process_data.py`：数据清洗与特征工程
- `LSTM.py`、`DFT.py`、`Trans.py`、`A_LSTM_test.py`、`A_Transformers_test.py`：三种模型的实现与训练
- `outcome.py`：结果汇总与对比分析

## 结果输出

各模型的预测结果、损失曲线等会保存在对应的`*_result/`和`middle_data/`目录下。 
