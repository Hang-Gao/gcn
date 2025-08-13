import pandas as pd
import re
from typing import Dict, List, Tuple, Any
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CSVDataParser:
    """
    CSV数据解析器，专门处理alpha表达式数据格式
    
    CSV格式：<name>::<表达式;num 数据类型1 编号1 数据类型2 编号2 ...>::<其他参数>
    特征编号映射：
    TCFBaseClean (1-7): 1→Ret, 2→open, 3→high, 4→low, 5→close, 6→vol, 7→oi
    TCFBidAskPrice (1-12): 1→mean_a_minus_b, 2→spread_indicator, ..., 12→total_volume
    """
    
    def __init__(self):
        # 特征编号到名称的映射
        # 每种数据类型都有自己的编号体系
        self.tcf_base_clean_mapping = {
            1: "Ret",     # 收益率
            2: "open",    # 开盘价  
            3: "high",    # 最高价
            4: "low",     # 最低价
            5: "close",   # 收盘价
            6: "vol",     # 成交量
            7: "oi"       # 持仓量
        }
        
        self.tcf_bid_ask_price_mapping = {
            1: "mean_a_minus_b",           # mean(a-b)
            2: "spread_indicator",         # (2p-a-b)/(a-b), mean
            3: "vol_weighted_spread",      # (2p-a-b)/(a-b) * vol, sum  
            4: "vwap",                     # vwap (sum(p * vol)/sum(vol))
            5: "price_volatility",         # p vola(mean(abs(mean(p) -p)))
            6: "spread_volatility",        # (a-b) vola
            7: "ask_volatility",           # a vola
            8: "bid_volatility",           # b vola
            9: "ask_delta_vol",            # delta(a) * va, sum
            10: "bid_delta_vol",           # delta(b) * vb, sum
            11: "delta_vol_diff",          # (delta(a)-delta(b)) * vol, sum
            12: "total_volume"             # sum(vol)
        }
    
    def load_csv_data(self, file_path: str) -> pd.DataFrame:
        """
        加载CSV文件
        
        Args:
            file_path: CSV文件路径
            
        Returns:
            pandas DataFrame
        """
        try:
            df = pd.read_csv(file_path)
            logger.info(f"成功加载CSV文件: {file_path}, 数据形状: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"加载CSV文件失败: {file_path}, 错误: {str(e)}")
            raise
    
    def parse_expression_string(self, expr_string: str) -> Dict[str, Any]:
        """
        解析复杂的表达式字符串
        
        格式: <name>::<表达式;num 数据编号映射>::<其他参数>
        例子: "SST_TM40Test1_181_27675::-sub(ts_norm(x2,1110),ts_norm(x1,1110));2 TCFBaseClean 5 TCFBaseClean 2::ma_1 225::..."
        
        Args:
            expr_string: 完整的表达式字符串
            
        Returns:
            解析后的字典，包含name, expression, variable_mapping, num_variables等
        """
        try:
            # 按::分割主要部分
            parts = expr_string.split("::")
            if len(parts) < 2:
                raise ValueError(f"表达式格式错误，缺少必要的::分隔符: {expr_string}")
            
            name = parts[0]
            expr_part = parts[1]
            
            # 解析表达式部分：expression;num data_mappings
            if ';' not in expr_part:
                raise ValueError(f"表达式部分缺少;分隔符: {expr_part}")
            
            expr_split = expr_part.split(';', 1)
            expression = expr_split[0]
            mapping_part = expr_split[1]
            
            # 解析变量映射信息
            variable_mapping, num_variables = self._parse_variable_mapping(mapping_part)
            
            result = {
                'name': name,
                'expression': expression,
                'variable_mapping': variable_mapping,
                'num_variables': num_variables,
                'raw_string': expr_string
            }
            
            return result
            
        except Exception as e:
            logger.error(f"解析表达式字符串失败: {expr_string[:100]}..., 错误: {str(e)}")
            raise
    
    def _parse_variable_mapping(self, mapping_part: str) -> Tuple[Dict[str, str], int]:
        """
        解析变量映射部分
        
        格式: "num data1_name data1_id data2_name data2_id ..."
        例子: "2 TCFBaseClean 5 TCFBaseClean 2"
        
        Args:
            mapping_part: 映射部分字符串
            
        Returns:
            (variable_mapping, num_variables)
            variable_mapping: {x1: 'close', x2: 'open', ...}
            num_variables: 变量数量
        """
        parts = mapping_part.strip().split()
        
        if len(parts) < 1:
            raise ValueError(f"映射部分格式错误: {mapping_part}")
        
        try:
            num_variables = int(parts[0])
        except ValueError:
            raise ValueError(f"无法解析变量数量: {parts[0]}")
        
        # 解析数据映射：data_name data_id 对
        variable_mapping = {}
        expected_parts = 1 + num_variables * 2  # num + (name id) * num_variables
        
        if len(parts) != expected_parts:
            # 有些情况下可能有额外的字段，我们只取需要的部分
            logger.warning(f"映射部分长度不匹配，期望{expected_parts}，实际{len(parts)}: {mapping_part}")
        
        # 提取变量映射
        # 关键：x1,x2,x3的含义完全由数据编号动态决定，每个表达式都不同！
        # 例如：表达式A中x1可能=close(5)，表达式B中x1可能=low(4)
        # 格式：num data1_name data1_id data2_name data2_id ...
        # 映射：x1→第1个ID, x2→第2个ID, x3→第3个ID（按顺序）
        for i in range(num_variables):
            var_name = f"x{i+1}"
            
            # 计算对应的数据类型和ID位置：每个变量对应 (数据名称, 数据ID)
            # 跳过num，然后每个变量对应2个元素：名称+ID
            name_index = 1 + i * 2      # 数据类型名称索引
            id_index = 1 + i * 2 + 1    # 数据ID索引
            
            if name_index < len(parts) and id_index < len(parts):
                try:
                    data_type = parts[name_index]  # TCFBaseClean 或 TCFBidAskPrice
                    data_id = int(parts[id_index])
                    
                    # 根据数据类型选择相应的映射字典
                    if data_type == "TCFBaseClean":
                        mapping_dict = self.tcf_base_clean_mapping
                    elif data_type == "TCFBidAskPrice":
                        mapping_dict = self.tcf_bid_ask_price_mapping
                    else:
                        logger.warning(f"未知的数据类型: {data_type} for 变量 {var_name}")
                        variable_mapping[var_name] = f"unknown_{data_type}_{data_id}"
                        continue
                    
                    if data_id in mapping_dict:
                        feature_name = mapping_dict[data_id]
                        variable_mapping[var_name] = feature_name
                        logger.debug(f"变量映射: {var_name} -> {feature_name} ({data_type}:{data_id})")
                    else:
                        logger.warning(f"未知的特征编号: {data_id} for 数据类型 {data_type}, 变量 {var_name}")
                        variable_mapping[var_name] = f"unknown_{data_type}_{data_id}"
                except ValueError:
                    logger.warning(f"无法解析数据ID: {parts[id_index]} for 变量 {var_name}")
                    variable_mapping[var_name] = f"unknown_{parts[id_index]}"
            else:
                logger.warning(f"映射部分数据不足，变量{var_name}使用默认值")
                variable_mapping[var_name] = "unknown"
        
        return variable_mapping, num_variables
    
    def parse_dataset(self, file_path: str) -> List[Dict[str, Any]]:
        """
        解析整个数据集
        
        Args:
            file_path: CSV文件路径
            
        Returns:
            解析后的数据列表，每个元素包含expression_info和sharpe值
        """
        df = self.load_csv_data(file_path)
        
        if 'sExpr' not in df.columns or 'Sharpe' not in df.columns:
            raise ValueError("CSV文件必须包含sExpr和Sharpe列")
        
        parsed_data = []
        failed_count = 0
        
        for idx, row in df.iterrows():
            try:
                expr_info = self.parse_expression_string(row['sExpr'])
                
                data_point = {
                    'expression_info': expr_info,
                    'sharpe': float(row['Sharpe']),
                    'row_index': idx
                }
                
                parsed_data.append(data_point)
                
            except Exception as e:
                failed_count += 1
                logger.warning(f"第{idx}行解析失败: {str(e)}")
                continue
        
        logger.info(f"数据集解析完成: 成功{len(parsed_data)}条, 失败{failed_count}条")
        
        return parsed_data
    
    def get_dataset_statistics(self, parsed_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        获取数据集统计信息
        
        Args:
            parsed_data: 解析后的数据
            
        Returns:
            统计信息字典
        """
        if not parsed_data:
            return {}
        
        sharpe_values = [d['sharpe'] for d in parsed_data]
        expressions = [d['expression_info']['expression'] for d in parsed_data]
        
        # 统计操作符使用频率
        operator_counts = {}
        for expr in expressions:
            # 简单的操作符提取（基于已知的16个操作符）
            operators = ['ts_detrend', 'ts_avg', 'abs', 'div', 'mul', 'mul_p', 'power_', 
                        'rsi', 'sign', 'sub', 'ts_corr', 'ts_norm', 'ts_ret', 'ts_skew', 
                        'ts_std', 'ts_sub_mean']
            
            for op in operators:
                count = expr.count(op)
                operator_counts[op] = operator_counts.get(op, 0) + count
        
        stats = {
            'total_samples': len(parsed_data),
            'sharpe_min': min(sharpe_values),
            'sharpe_max': max(sharpe_values),
            'sharpe_mean': sum(sharpe_values) / len(sharpe_values),
            'avg_expression_length': sum(len(expr) for expr in expressions) / len(expressions),
            'operator_counts': operator_counts
        }
        
        return stats
    
    def validate_data(self, parsed_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        验证解析后的数据质量
        
        Args:
            parsed_data: 解析后的数据
            
        Returns:
            验证结果字典
        """
        validation_results = {
            'total_samples': len(parsed_data),
            'valid_samples': 0,
            'invalid_samples': 0,
            'validation_errors': []
        }
        
        for idx, data_point in enumerate(parsed_data):
            try:
                # 验证Sharpe值
                sharpe = data_point['sharpe']
                if not isinstance(sharpe, (int, float)) or pd.isna(sharpe):
                    validation_results['validation_errors'].append(
                        f"样本{idx}: Sharpe值无效 ({sharpe})"
                    )
                    validation_results['invalid_samples'] += 1
                    continue
                
                # 验证表达式信息
                expr_info = data_point['expression_info']
                if not expr_info.get('expression'):
                    validation_results['validation_errors'].append(
                        f"样本{idx}: 表达式为空"
                    )
                    validation_results['invalid_samples'] += 1
                    continue
                
                # 验证变量映射
                if not expr_info.get('variable_mapping'):
                    validation_results['validation_errors'].append(
                        f"样本{idx}: 变量映射为空"
                    )
                    validation_results['invalid_samples'] += 1
                    continue
                
                # 验证变量数量
                num_vars = expr_info.get('num_variables', 0)
                if num_vars <= 0 or num_vars > 10:  # 合理范围检查
                    validation_results['validation_errors'].append(
                        f"样本{idx}: 变量数量异常 ({num_vars})"
                    )
                    validation_results['invalid_samples'] += 1
                    continue
                
                validation_results['valid_samples'] += 1
                
            except Exception as e:
                validation_results['validation_errors'].append(
                    f"样本{idx}: 验证过程中出错 - {str(e)}"
                )
                validation_results['invalid_samples'] += 1
        
        # 计算验证统计
        validation_results['validation_rate'] = (
            validation_results['valid_samples'] / validation_results['total_samples']
            if validation_results['total_samples'] > 0 else 0
        )
        
        logger.info(f"数据验证完成: 有效{validation_results['valid_samples']}条, "
                   f"无效{validation_results['invalid_samples']}条, "
                   f"验证率{validation_results['validation_rate']:.2%}")
        
        return validation_results
    
    def clean_data(self, parsed_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        清洗数据，移除无效样本
        
        Args:
            parsed_data: 原始解析数据
            
        Returns:
            清洗后的数据
        """
        cleaned_data = []
        
        for data_point in parsed_data:
            try:
                # 检查必要字段
                if (data_point.get('sharpe') is not None and 
                    data_point.get('expression_info') and
                    data_point['expression_info'].get('expression') and
                    data_point['expression_info'].get('variable_mapping')):
                    
                    # 检查变量映射中是否包含未知的数据类型（如TCFTopBook等）
                    variable_mapping = data_point['expression_info']['variable_mapping']
                    has_unknown_data_type = any(
                        feature.startswith('unknown_') for feature in variable_mapping.values()
                    )
                    
                    if has_unknown_data_type:
                        logger.debug(f"过滤包含未知数据类型的表达式: {variable_mapping}")
                        continue
                    
                    # 检查Sharpe值范围（基于文档中的范围0.0801-0.2104）
                    sharpe = data_point['sharpe']
                    if -1.0 <= sharpe <= 1.0:  # 扩大范围以包含可能的负值
                        cleaned_data.append(data_point)
                    else:
                        logger.warning(f"Sharpe值超出合理范围: {sharpe}")
                        
            except Exception as e:
                logger.warning(f"清洗数据时出错: {str(e)}")
                continue
        
        logger.info(f"数据清洗完成: 原始{len(parsed_data)}条, 清洗后{len(cleaned_data)}条")
        
        return cleaned_data