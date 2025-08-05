# Alpha表达式GCN评估器

基于图卷积神经网络(GCN)的Alpha表达式Sharpe值预测系统。

## 项目概述

本项目实现了一个端到端的深度学习系统，用于预测金融alpha表达式的Sharpe比率。核心特点包括：

- **动态变量映射**: 深度理解alpha表达式中x1、x2、x3变量的动态映射机制
- **图神经网络**: 将复杂的alpha表达式转换为图结构，使用GCN进行预测
- **完整流程**: 数据加载→表达式解析→图转换→模型训练→性能评估

## 环境要求

- Python 3.7+
- PyTorch
- PyTorch Geometric
- pandas, numpy, matplotlib

## 快速开始

### 1. 基本运行
```bash
# 使用默认参数运行
python main.py
```

### 2. 自定义参数运行
```bash
# 自定义训练参数
python main.py --epochs 50 --batch-size 32 --hidden-dim 128
```

## 运行参数说明

### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--config` | str | None | 配置文件路径 |
| `--train-file` | str | `alpha_perf_IS.csv` | 训练数据文件 |
| `--test-file` | str | `alpha_perf_OS.csv` | 测试数据文件 |
| `--epochs` | int | **100** | 训练轮数 |
| `--batch-size` | int | **32** | 批次大小 |
| `--hidden-dim` | int | **128** | GCN隐藏层维度 |
| `--device` | str | **cpu** | 计算设备 (cpu/cuda) |
| `--log-level` | str | **INFO** | 日志级别 (DEBUG/INFO/WARNING/ERROR) |

### 默认配置详情

当使用默认参数时，完整的配置如下：

```python
默认配置 = {
    'data': {
        'train_file': 'alpha_perf_IS.csv',      # 训练数据文件
        'test_file': 'alpha_perf_OS.csv'        # 测试数据文件
    },
    'model': {
        'hidden_dim': 128,                      # GCN隐藏层维度
        'num_layers': 3,                        # GCN网络层数
        'dropout': 0.2                          # Dropout率
    },
    'training': {
        'epochs': 100,                          # 训练轮数
        'batch_size': 32,                       # 批次大小
        'learning_rate': 0.001,                 # 学习率
        'weight_decay': 1e-5,                   # 权重衰减
        'scheduler': 'ReduceLROnPlateau',       # 学习率调度器
        'val_ratio': 0.1                        # 验证集比例（从训练集中分出）
    },
    'output': {
        'results_dir': 'results',               # 结果输出目录
        'log_level': 'INFO'                     # 日志级别
    },
    'device': 'cpu'                             # 计算设备
}
```

## 推荐参数配置

### 快速测试（约30秒）
```bash
python main.py --epochs 10 --batch-size 16 --hidden-dim 64
```

### 标准训练（约2-3分钟）
```bash
python main.py --epochs 50 --batch-size 32 --hidden-dim 128
```

### 完整训练（约5-10分钟）
```bash
python main.py --epochs 100 --batch-size 32 --hidden-dim 256
```

### 调试模式
```bash
python main.py --epochs 5 --batch-size 8 --log-level DEBUG
```

## 预期运行结果

### 数据加载阶段
```
开始加载数据...
训练数据加载完成: 539 个样本
测试数据加载完成: 539 个样本
```

### 模型训练阶段
```
模型初始化完成:
  输入维度: 31
  隐藏维度: 128
  网络层数: 3
  总参数量: 50,000+
```

### 性能评估结果
```
模型评估结果:
  样本数量: 539
  MSE: 0.005430
  MAE: 0.054570
  RMSE: 0.073670
  R²: -0.268000
  相关性: 0.120000
```

## 输出文件

运行完成后，会在`results/`目录下生成：

```
results/
├── logs/                           # 训练日志
│   └── main_YYYYMMDD_HHMMSS.log
├── model_checkpoints/              # 模型保存
│   └── alpha_gcn_model_YYYYMMDD_HHMMSS.pth
├── visualizations/                 # 可视化图表
│   ├── training_curves_YYYYMMDD_HHMMSS.png
│   └── predictions_vs_true_YYYYMMDD_HHMMSS.png
├── full_results_YYYYMMDD_HHMMSS.json  # 完整结果
└── data_validation_report.json        # 数据验证报告
```

## 数据格式说明

### CSV数据格式
```
<name>::<表达式;num 数据1名称 数据1编号 数据2名称 数据2编号...>::<均值平滑>::<指数平滑>::<中性化方法>::<标准化方法>::<InSample起始时间>::<OutSample起始时间>
```

### alpha表达式中变量说明：
变量：x1、x2、x3 - 注意：变量数量由num参数决定，x1/x2/x3的具体含义由表达式后的数据编号动态映射到7个特征(1→Ret, 2→open, 3→high, 4→low, 5→close, 6→vol, 7→oi)

### 特征数据映射
| 编号 | 特征名称 | 说明 |
|------|----------|------|
| 1 | Ret | 收益率 |
| 2 | open | 开盘价 |
| 3 | high | 最高价 |
| 4 | low | 最低价 |
| 5 | close | 收盘价 |
| 6 | vol | 成交量 |
| 7 | oi | 持仓量 |

### 支持的操作符（16个）
- `abs` - 绝对值
- `div` - 除法
- `mul` - 乘法
- `mul_p` - 乘法（第二个变量为具体数字）
- `power_` - 幂运算
- `rsi` - 相对强弱指标
- `sign` - 取符号量
- `sub` - 减法
- `ts_avg` - 时间序列平均值
- `ts_corr` - 时间序列相关性
- `ts_detrend` - 时间序列去趋势
- `ts_norm` - 时间序列标准化
- `ts_ret` - 时间序列收益率
- `ts_skew` - 时间序列偏度
- `ts_std` - 时间序列标准差
- `ts_sub_mean` - ts_norm的分子部分

## 故障排除

### 常见问题

1. **环境问题**: 确保已激活gcn虚拟环境
2. **内存不足**: 减小batch_size或hidden_dim
3. **训练过慢**: 减少epochs数量或使用GPU加速
4. **模型不收敛**: 调整learning_rate或增加训练数据

### 调试技巧

```bash
# 使用调试模式查看详细信息
python main.py --log-level DEBUG --epochs 5

# 使用小批次快速验证
python main.py --batch-size 8 --epochs 3
```

## 项目结构

```
gcn/
├── main.py                    # 主程序入口
├── data/                      # 数据处理模块
│   ├── csv_parser.py         # CSV数据解析器
│   ├── dataset.py            # PyTorch数据集类
│   └── preprocessing.py      # 数据预处理
├── parser/                   # 表达式解析模块
│   ├── expression_parser.py  # 表达式解析器
│   ├── ast_nodes.py         # AST节点定义
│   └── operators.py         # 操作符定义
├── model/                    # 模型定义模块
│   ├── gcn_model.py         # GCN模型定义
│   ├── graph_builder.py     # 图构建器
│   └── trainer.py           # 训练评估逻辑
├── utils/                    # 工具模块
│   ├── metrics.py           # 评估指标
│   └── visualization.py     # 可视化工具
├── results/                  # 结果输出目录
├── alpha_perf_IS.csv        # 训练数据
├── alpha_perf_OS.csv        # 测试数据
└── README.md                # 本文档
```
