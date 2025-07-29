import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler

logger = logging.getLogger(__name__)

class AlphaDataPreprocessor:
    """
    Alpha表达式数据预处理器
    
    负责对数据进行标准化、归一化等预处理操作
    """
    
    def __init__(self):
        self.sharpe_scaler = None
        self.fitted = False
        
    def fit_sharpe_scaler(self, sharpe_values: List[float], scaler_type: str = 'standard') -> None:
        """
        训练Sharpe值的缩放器
        
        Args:
            sharpe_values: Sharpe值列表
            scaler_type: 缩放器类型，'standard' 或 'minmax'
        """
        sharpe_array = np.array(sharpe_values).reshape(-1, 1)
        
        if scaler_type == 'standard':
            self.sharpe_scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.sharpe_scaler = MinMaxScaler()
        else:
            raise ValueError(f"不支持的缩放器类型: {scaler_type}")
        
        self.sharpe_scaler.fit(sharpe_array)
        self.fitted = True
        
        logger.info(f"Sharpe值缩放器训练完成，类型: {scaler_type}")
        
        if hasattr(self.sharpe_scaler, 'mean_'):
            logger.info(f"标准化参数 - 均值: {self.sharpe_scaler.mean_[0]:.4f}, "
                       f"标准差: {self.sharpe_scaler.scale_[0]:.4f}")
        elif hasattr(self.sharpe_scaler, 'data_min_'):
            logger.info(f"归一化参数 - 最小值: {self.sharpe_scaler.data_min_[0]:.4f}, "
                       f"最大值: {self.sharpe_scaler.data_max_[0]:.4f}")
    
    def transform_sharpe(self, sharpe_values: List[float]) -> np.ndarray:
        """
        转换Sharpe值
        
        Args:
            sharpe_values: 待转换的Sharpe值列表
            
        Returns:
            转换后的numpy数组
        """
        if not self.fitted:
            raise ValueError("缩放器尚未训练，请先调用fit_sharpe_scaler()")
        
        sharpe_array = np.array(sharpe_values).reshape(-1, 1)
        return self.sharpe_scaler.transform(sharpe_array).flatten()
    
    def inverse_transform_sharpe(self, scaled_sharpe: np.ndarray) -> np.ndarray:
        """
        反向转换Sharpe值
        
        Args:
            scaled_sharpe: 已缩放的Sharpe值
            
        Returns:
            原始尺度的Sharpe值
        """
        if not self.fitted:
            raise ValueError("缩放器尚未训练，请先调用fit_sharpe_scaler()")
        
        if scaled_sharpe.ndim == 1:
            scaled_sharpe = scaled_sharpe.reshape(-1, 1)
        
        return self.sharpe_scaler.inverse_transform(scaled_sharpe).flatten()
    
    def get_sharpe_statistics(self, sharpe_values: List[float]) -> Dict[str, float]:
        """
        获取Sharpe值统计信息
        
        Args:
            sharpe_values: Sharpe值列表
            
        Returns:
            统计信息字典
        """
        sharpe_array = np.array(sharpe_values)
        
        return {
            'count': len(sharpe_array),
            'mean': float(np.mean(sharpe_array)),
            'std': float(np.std(sharpe_array)),
            'min': float(np.min(sharpe_array)),
            'max': float(np.max(sharpe_array)),
            'median': float(np.median(sharpe_array)),
            'q25': float(np.percentile(sharpe_array, 25)),
            'q75': float(np.percentile(sharpe_array, 75))
        }


class ExpressionPreprocessor:
    """
    表达式预处理器
    
    负责表达式文本的标准化和清理
    """
    
    def __init__(self):
        # 已知的16个操作符
        self.known_operators = [
            'ts_detrend', 'ts_avg', 'abs', 'div', 'mul', 'mul_p', 'power_',
            'rsi', 'sign', 'sub', 'ts_corr', 'ts_norm', 'ts_ret', 'ts_skew',
            'ts_std', 'ts_sub_mean'
        ]
        
        # 操作符映射（处理同义词）
        self.operator_mapping = {
            'mul_p': 'mul',  # mul_p等价于mul
        }
    
    def normalize_expression(self, expression: str) -> str:
        """
        标准化表达式
        
        Args:
            expression: 原始表达式字符串
            
        Returns:
            标准化后的表达式
        """
        # 去除多余空格
        normalized = ' '.join(expression.split())
        
        # 应用操作符映射
        for old_op, new_op in self.operator_mapping.items():
            normalized = normalized.replace(old_op, new_op)
        
        return normalized
    
    def extract_operators(self, expression: str) -> List[str]:
        """
        从表达式中提取操作符
        
        Args:
            expression: 表达式字符串
            
        Returns:
            操作符列表
        """
        operators = []
        
        for op in self.known_operators:
            if op in expression:
                # 计算该操作符在表达式中的出现次数
                count = expression.count(op)
                operators.extend([op] * count)
        
        return operators
    
    def extract_variables(self, expression: str) -> List[str]:
        """
        从表达式中提取变量
        
        Args:
            expression: 表达式字符串
            
        Returns:
            变量列表
        """
        import re
        
        # 查找所有x+数字的模式
        variable_pattern = r'x\d+'
        variables = re.findall(variable_pattern, expression)
        
        return list(set(variables))  # 去重
    
    def extract_constants(self, expression: str) -> List[float]:
        """
        从表达式中提取数值常量
        
        Args:
            expression: 表达式字符串
            
        Returns:
            常量列表
        """
        import re
        
        # 查找数字（包括小数和负数）
        number_pattern = r'-?\d+\.?\d*'
        matches = re.findall(number_pattern, expression)
        
        constants = []
        for match in matches:
            try:
                # 过滤掉变量中的数字（如x1中的1）
                if not any(f'x{match}' in expression for _ in [1]):
                    constants.append(float(match))
            except ValueError:
                continue
        
        return constants
    
    def get_expression_complexity(self, expression: str) -> Dict[str, Any]:
        """
        计算表达式复杂度指标
        
        Args:
            expression: 表达式字符串
            
        Returns:
            复杂度指标字典
        """
        operators = self.extract_operators(expression)
        variables = self.extract_variables(expression)
        constants = self.extract_constants(expression)
        
        # 计算嵌套深度（通过括号配对）
        nesting_depth = self._calculate_nesting_depth(expression)
        
        return {
            'length': len(expression),
            'operator_count': len(operators),
            'unique_operators': len(set(operators)),
            'variable_count': len(variables),
            'constant_count': len(constants),
            'nesting_depth': nesting_depth,
            'operators': operators,
            'variables': variables,
            'constants': constants
        }
    
    def _calculate_nesting_depth(self, expression: str) -> int:
        """
        计算表达式的嵌套深度
        
        Args:
            expression: 表达式字符串
            
        Returns:
            最大嵌套深度
        """
        max_depth = 0
        current_depth = 0
        
        for char in expression:
            if char == '(':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == ')':
                current_depth -= 1
        
        return max_depth


class DatasetPreprocessor:
    """
    数据集级别的预处理器
    
    整合所有预处理操作
    """
    
    def __init__(self):
        self.alpha_preprocessor = AlphaDataPreprocessor()
        self.expr_preprocessor = ExpressionPreprocessor()
        
    def preprocess_dataset(self, dataset, fit_scalers: bool = True) -> Dict[str, Any]:
        """
        预处理整个数据集
        
        Args:
            dataset: AlphaExpressionDataset实例
            fit_scalers: 是否训练缩放器（通常只在训练集上设为True）
            
        Returns:
            预处理统计信息
        """
        # 提取所有Sharpe值
        sharpe_values = [item['sharpe'] for item in dataset.data]
        
        # 如果需要，训练Sharpe缩放器
        if fit_scalers:
            self.alpha_preprocessor.fit_sharpe_scaler(sharpe_values)
        
        # 计算表达式复杂度统计
        expression_stats = self._analyze_expressions(dataset)
        
        # 计算Sharpe统计
        sharpe_stats = self.alpha_preprocessor.get_sharpe_statistics(sharpe_values)
        
        preprocessing_info = {
            'dataset_size': len(dataset),
            'sharpe_stats': sharpe_stats,
            'expression_stats': expression_stats,
            'scaler_fitted': self.alpha_preprocessor.fitted
        }
        
        logger.info(f"数据集预处理完成，大小: {len(dataset)}")
        logger.info(f"Sharpe统计 - 均值: {sharpe_stats['mean']:.4f}, "
                   f"标准差: {sharpe_stats['std']:.4f}, "
                   f"范围: [{sharpe_stats['min']:.4f}, {sharpe_stats['max']:.4f}]")
        
        return preprocessing_info
    
    def _analyze_expressions(self, dataset) -> Dict[str, Any]:
        """
        分析数据集中的表达式
        
        Args:
            dataset: AlphaExpressionDataset实例
            
        Returns:
            表达式分析结果
        """
        all_operators = []
        all_variables = []
        complexities = []
        
        for item in dataset.data:
            expression = item['expression_info']['expression']
            
            # 标准化表达式
            normalized_expr = self.expr_preprocessor.normalize_expression(expression)
            
            # 提取特征
            complexity = self.expr_preprocessor.get_expression_complexity(normalized_expr)
            complexities.append(complexity)
            
            all_operators.extend(complexity['operators'])
            all_variables.extend(complexity['variables'])
        
        # 统计操作符使用频率
        operator_counts = {}
        for op in all_operators:
            operator_counts[op] = operator_counts.get(op, 0) + 1
        
        # 统计变量使用频率
        variable_counts = {}
        for var in all_variables:
            variable_counts[var] = variable_counts.get(var, 0) + 1
        
        # 计算复杂度统计
        lengths = [c['length'] for c in complexities]
        depths = [c['nesting_depth'] for c in complexities]
        op_counts = [c['operator_count'] for c in complexities]
        
        return {
            'total_expressions': len(complexities),
            'operator_frequency': operator_counts,
            'variable_frequency': variable_counts,
            'complexity_stats': {
                'avg_length': np.mean(lengths),
                'avg_nesting_depth': np.mean(depths),
                'avg_operator_count': np.mean(op_counts),
                'max_length': max(lengths),
                'max_nesting_depth': max(depths),
                'max_operator_count': max(op_counts)
            }
        }
    
    def transform_for_training(self, dataset, normalize_sharpe: bool = True):
        """
        为训练转换数据集
        
        Args:
            dataset: AlphaExpressionDataset实例
            normalize_sharpe: 是否标准化Sharpe值
            
        Returns:
            转换后的数据
        """
        if normalize_sharpe and not self.alpha_preprocessor.fitted:
            raise ValueError("Sharpe缩放器尚未训练，请先调用preprocess_dataset(fit_scalers=True)")
        
        transformed_data = []
        
        for item in dataset.data:
            # 复制原始项
            transformed_item = item.copy()
            
            # 可选择性转换Sharpe值
            if normalize_sharpe:
                original_sharpe = item['sharpe']
                scaled_sharpe = self.alpha_preprocessor.transform_sharpe([original_sharpe])[0]
                transformed_item['sharpe'] = scaled_sharpe
                transformed_item['original_sharpe'] = original_sharpe
            
            # 标准化表达式
            expr_info = transformed_item['expression_info'].copy()
            expr_info['expression'] = self.expr_preprocessor.normalize_expression(
                expr_info['expression']
            )
            transformed_item['expression_info'] = expr_info
            
            transformed_data.append(transformed_item)
        
        logger.info(f"数据转换完成，样本数量: {len(transformed_data)}")
        
        return transformed_data