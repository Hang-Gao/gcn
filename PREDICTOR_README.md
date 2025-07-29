# Alpha表达式Sharpe值预测器

这是一个独立的预测程序，使用训练好的GCN模型预测alpha表达式的Sharpe值。

## 功能特点

- **智能模型选择**：自动选择最新模型，或手动指定模型
- **灵活输入方式**：支持单个表达式、多个表达式、文件批量处理
- **多种文件格式**：支持TXT和CSV文件
- **多种输出格式**：简洁、详细、JSON格式
- **交互式模式**：支持实时交互预测
- **批量处理**：高效的批量预测能力

## 使用方法

### 1. 查看可用模型
```bash
python predict.py --list-models
```

### 2. 预测单个表达式
```bash
python predict.py --expression "abs(div(x1,x2));2 TCFBaseClean 2 TCFBaseClean 5"
```

### 3. 预测多个表达式
```bash
python predict.py --expressions "abs(x1);1 TCFBaseClean 5" "sub(x1,x2);2 TCFBaseClean 1 TCFBaseClean 2"
```

### 4. 从文件批量预测

**TXT文件格式：**
```
abs(div(x1,x2));2 TCFBaseClean 2 TCFBaseClean 5
sub(x1,x2);2 TCFBaseClean 1 TCFBaseClean 2
ts_avg(x1,10);1 TCFBaseClean 5
```

**CSV文件格式：**
```csv
expression
abs(div(x1,x2));2 TCFBaseClean 2 TCFBaseClean 5
sub(x1,x2);2 TCFBaseClean 1 TCFBaseClean 2
```

**使用命令：**
```bash
# 从TXT文件预测
python predict.py --file expressions.txt

# 从CSV文件预测
python predict.py --file expressions.csv

# 指定CSV列
python predict.py --file data.csv --column expression
```

### 5. 交互式模式
```bash
python predict.py --interactive
```

### 6. 不同输出格式

**简洁格式（仅显示数值）：**
```bash
python predict.py --expression "abs(x1);1 TCFBaseClean 5" --format simple
# 输出: 0.118545
```

**详细格式（默认）：**
```bash
python predict.py --expression "abs(x1);1 TCFBaseClean 5" --format detailed
# 输出包含表达式、预测值、变量映射等详细信息
```

**JSON格式：**
```bash
python predict.py --expression "abs(x1);1 TCFBaseClean 5" --format json
# 输出结构化JSON格式
```

### 7. 保存结果到文件
```bash
python predict.py --file input.txt --output results.csv
```

### 8. 指定模型和设备
```bash
# 使用特定模型
python predict.py --model 2 --expression "abs(x1);1 TCFBaseClean 5"

# 指定计算设备
python predict.py --device cpu --expression "abs(x1);1 TCFBaseClean 5"
```

## 表达式格式说明

支持两种格式：

1. **简化格式（推荐）：**
   ```
   表达式;num 数据1名称 数据1编号 数据2名称 数据2编号...
   ```
   例如：`abs(div(x1,x2));2 TCFBaseClean 2 TCFBaseClean 5`

2. **完整CSV格式：**
   ```
   <name>::<表达式;num 数据1名称 数据1编号 数据2名称 数据2编号...>::<均值平滑>::<指数平滑>::<中性化方法>::<标准化方法>::<InSample起始时间>::<OutSample起始时间>
   ```
   例如：`SST_TM40Test1_181_27675::-sub(ts_norm(x2,1110),ts_norm(x1,1110));2 TCFBaseClean 5 TCFBaseClean 2::ma_1 225::ema1_1 0.01::no_neu::general_nor::20110104::20180102`

## 变量映射说明

- **变量**：x1, x2, x3等（数量由num参数决定）
- **特征编号映射**：
  - 1 → Ret (收益率)
  - 2 → open (开盘价) 
  - 3 → high (最高价)
  - 4 → low (最低价)
  - 5 → close (收盘价)
  - 6 → vol (成交量)
  - 7 → oi (持仓量)

例如：`abs(div(x1,x2));2 TCFBaseClean 2 TCFBaseClean 5`
- x1 映射到特征2（开盘价）
- x2 映射到特征5（收盘价）

## 支持的操作符

包含16个主要操作符：
- **时间序列操作符**：ts_avg, ts_norm, ts_detrend, ts_ret, ts_corr, ts_std, ts_skew, ts_sub_mean
- **数学操作符**：abs, div, mul, sub, power_, sign
- **技术指标**：rsi

## 示例输出

```json
{
  "expression": "abs(div(x1,x2));2 TCFBaseClean 2 TCFBaseClean 5",
  "predicted_sharpe": 0.164459,
  "parsed_expression": "abs(div(x1,x2))",
  "variable_mapping": {
    "x1": "open",
    "x2": "close"
  }
}
```

## 性能说明

- **模型参数**：约18,881个参数
- **支持设备**：CPU, CUDA, MPS
- **批量处理**：支持高效的批量预测
- **内存占用**：轻量级，适合生产环境部署

## 常见问题

1. **Q: 如何选择模型？**
   A: 默认自动选择最新模型，也可以用--model参数指定

2. **Q: 支持哪些文件格式？**
   A: 支持TXT和CSV文件，自动检测格式

3. **Q: 如何处理大量表达式？**
   A: 使用文件输入模式，支持高效的批量处理

4. **Q: 预测结果的范围是什么？**
   A: Sharpe值通常在0.08-0.21之间

## 技术架构

- **模型类型**：图卷积神经网络(GCN)
- **输入表示**：表达式抽象语法树转图结构
- **特征维度**：31维节点特征
- **隐藏层数**：3层GCN + 批归一化 + 残差连接