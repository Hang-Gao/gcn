from typing import Dict, List, Callable, Any, Optional
import logging

logger = logging.getLogger(__name__)

class OperatorDefinition:
    """
    操作符定义类
    
    定义每个操作符的基本属性和行为
    """
    
    def __init__(self, name: str, arity: int, category: str, description: str = ""):
        self.name = name                # 操作符名称
        self.arity = arity             # 参数个数 (-1表示可变参数)
        self.category = category       # 操作符类别
        self.description = description # 操作符描述
        
    def __repr__(self):
        return f"OperatorDefinition(name='{self.name}', arity={self.arity}, category='{self.category}')"


class OperatorRegistry:
    """
    操作符注册表
    
    管理所有已知的操作符定义
    """
    
    def __init__(self):
        self.operators: Dict[str, OperatorDefinition] = {}
        self._register_builtin_operators()
    
    def _register_builtin_operators(self):
        """注册内置的16个操作符"""
        
        # 时间序列操作符
        self.register(OperatorDefinition(
            name='ts_detrend',
            arity=2,
            category='time_series',
            description='时间序列去趋势：当前分钟数据减去n日内当前分钟的均值'
        ))
        
        self.register(OperatorDefinition(
            name='ts_avg',
            arity=2,
            category='time_series',
            description='时间序列平均值：计算过去n期的平均值'
        ))
        
        self.register(OperatorDefinition(
            name='ts_norm',
            arity=2,
            category='time_series',
            description='时间序列标准化：当前分钟数据减去过去t分钟均值，再除以过去t分钟标准差'
        ))
        
        self.register(OperatorDefinition(
            name='ts_std',
            arity=2,
            category='time_series',
            description='时间序列标准差：计算过去n期的标准差'
        ))
        
        self.register(OperatorDefinition(
            name='ts_ret',
            arity=2,
            category='time_series',
            description='时间序列收益率：当前分钟数据除t分钟之前数据再减1'
        ))
        
        self.register(OperatorDefinition(
            name='ts_corr',
            arity=3,
            category='time_series',
            description='时间序列相关性：计算两个序列在过去n期的相关系数'
        ))
        
        self.register(OperatorDefinition(
            name='ts_skew',
            arity=2,
            category='time_series',
            description='时间序列偏度：计算过去n期的偏度'
        ))
        
        self.register(OperatorDefinition(
            name='ts_sub_mean',
            arity=2,
            category='time_series',
            description='ts_norm的分子部分：当前分钟数据减去过去t分钟均值'
        ))
        
        # 基础数学操作符
        self.register(OperatorDefinition(
            name='abs',
            arity=1,
            category='math_basic',
            description='绝对值：取输入的绝对值'
        ))
        
        self.register(OperatorDefinition(
            name='div',
            arity=2,
            category='math_basic',
            description='除法：第一个参数除以第二个参数'
        ))
        
        self.register(OperatorDefinition(
            name='mul',
            arity=2,
            category='math_basic',
            description='乘法：两个参数相乘'
        ))
        
        self.register(OperatorDefinition(
            name='mul_p',
            arity=2,
            category='math_basic',
            description='乘法（等价于mul）：第二个变量为具体数字'
        ))
        
        self.register(OperatorDefinition(
            name='sub',
            arity=2,
            category='math_basic',
            description='减法：第一个参数减去第二个参数'
        ))
        
        self.register(OperatorDefinition(
            name='power_',
            arity=2,
            category='math_basic',
            description='幂运算：第一个参数的第二个参数次方'
        ))
        
        self.register(OperatorDefinition(
            name='sign',
            arity=1,
            category='math_basic',
            description='取符号量：值为±1或0'
        ))
        
        # 技术指标操作符
        self.register(OperatorDefinition(
            name='rsi',
            arity=2,
            category='technical_indicator',
            description='相对强弱指标：RSI技术指标'
        ))
        
        logger.info(f"已注册 {len(self.operators)} 个操作符")
    
    def register(self, operator_def: OperatorDefinition):
        """
        注册操作符
        
        Args:
            operator_def: 操作符定义
        """
        self.operators[operator_def.name] = operator_def
    
    def get_operator(self, name: str) -> Optional[OperatorDefinition]:
        """
        获取操作符定义
        
        Args:
            name: 操作符名称
            
        Returns:
            操作符定义，如果不存在返回None
        """
        return self.operators.get(name)
    
    def is_known_operator(self, name: str) -> bool:
        """
        检查是否为已知操作符
        
        Args:
            name: 操作符名称
            
        Returns:
            是否为已知操作符
        """
        return name in self.operators
    
    def get_operators_by_category(self, category: str) -> List[OperatorDefinition]:
        """
        根据类别获取操作符
        
        Args:
            category: 操作符类别
            
        Returns:
            该类别的操作符列表
        """
        return [op for op in self.operators.values() if op.category == category]
    
    def get_all_operators(self) -> List[OperatorDefinition]:
        """
        获取所有操作符
        
        Returns:
            所有操作符列表
        """
        return list(self.operators.values())
    
    def get_operator_names(self) -> List[str]:
        """
        获取所有操作符名称
        
        Returns:
            操作符名称列表
        """
        return list(self.operators.keys())
    
    def validate_operator_usage(self, name: str, arg_count: int) -> Dict[str, Any]:
        """
        验证操作符使用是否正确
        
        Args:
            name: 操作符名称
            arg_count: 实际参数个数
            
        Returns:
            验证结果字典
        """
        if not self.is_known_operator(name):
            return {
                'valid': False,
                'error': f"未知操作符: {name}",
                'suggestion': f"已知操作符: {', '.join(self.get_operator_names())}"
            }
        
        operator_def = self.get_operator(name)
        expected_arity = operator_def.arity
        
        # -1表示可变参数
        if expected_arity == -1:
            return {'valid': True}
        
        if arg_count != expected_arity:
            return {
                'valid': False,
                'error': f"操作符 {name} 期望 {expected_arity} 个参数，实际提供 {arg_count} 个",
                'expected_arity': expected_arity,
                'actual_arity': arg_count
            }
        
        return {'valid': True}
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取操作符注册表统计信息
        
        Returns:
            统计信息字典
        """
        categories = {}
        arities = {}
        
        for op in self.operators.values():
            # 按类别统计
            categories[op.category] = categories.get(op.category, 0) + 1
            
            # 按参数个数统计
            arity_key = str(op.arity) if op.arity >= 0 else "variable"
            arities[arity_key] = arities.get(arity_key, 0) + 1
        
        return {
            'total_operators': len(self.operators),
            'categories': categories,
            'arity_distribution': arities,
            'operator_names': sorted(self.get_operator_names())
        }


class OperatorSemantics:
    """
    操作符语义处理器
    
    处理操作符的特殊语义，如mul_p等价于mul等
    """
    
    def __init__(self):
        # 操作符别名映射
        self.operator_aliases = {
            'mul_p': 'mul'  # mul_p等价于mul
        }
        
        # 特殊操作符处理规则
        self.special_rules = {
            'ts_sub_mean': 'ts_norm_numerator'  # ts_sub_mean是ts_norm的分子部分
        }
    
    def normalize_operator(self, operator_name: str) -> str:
        """
        标准化操作符名称
        
        Args:
            operator_name: 原始操作符名称
            
        Returns:
            标准化后的操作符名称
        """
        return self.operator_aliases.get(operator_name, operator_name)
    
    def get_semantic_type(self, operator_name: str) -> str:
        """
        获取操作符的语义类型
        
        Args:
            operator_name: 操作符名称
            
        Returns:
            语义类型
        """
        normalized = self.normalize_operator(operator_name)
        
        if normalized.startswith('ts_'):
            return 'time_series'
        elif normalized in ['abs', 'div', 'mul', 'sub', 'power_', 'sign']:
            return 'math_basic'
        elif normalized in ['rsi']:
            return 'technical_indicator'
        else:
            return 'unknown'
    
    def is_equivalent(self, op1: str, op2: str) -> bool:
        """
        检查两个操作符是否等价
        
        Args:
            op1: 第一个操作符
            op2: 第二个操作符
            
        Returns:
            是否等价
        """
        return self.normalize_operator(op1) == self.normalize_operator(op2)
    
    def get_special_handling(self, operator_name: str) -> Optional[str]:
        """
        获取操作符的特殊处理规则
        
        Args:
            operator_name: 操作符名称
            
        Returns:
            特殊处理规则，如果没有返回None
        """
        return self.special_rules.get(operator_name)


# 全局操作符注册表实例
operator_registry = OperatorRegistry()
operator_semantics = OperatorSemantics()


def get_operator_info(operator_name: str) -> Dict[str, Any]:
    """
    获取操作符的完整信息
    
    Args:
        operator_name: 操作符名称
        
    Returns:
        操作符信息字典
    """
    normalized_name = operator_semantics.normalize_operator(operator_name)
    operator_def = operator_registry.get_operator(normalized_name)
    
    if operator_def is None:
        return {
            'name': operator_name,
            'normalized_name': normalized_name,
            'exists': False,
            'semantic_type': operator_semantics.get_semantic_type(operator_name)
        }
    
    return {
        'name': operator_name,
        'normalized_name': normalized_name,
        'exists': True,
        'arity': operator_def.arity,
        'category': operator_def.category,
        'description': operator_def.description,
        'semantic_type': operator_semantics.get_semantic_type(operator_name),
        'special_handling': operator_semantics.get_special_handling(operator_name)
    }


def validate_expression_operators(operators: List[str]) -> Dict[str, Any]:
    """
    验证表达式中的所有操作符
    
    Args:
        operators: 操作符列表
        
    Returns:
        验证结果字典
    """
    validation_results = {
        'total_operators': len(operators),
        'valid_operators': 0,
        'invalid_operators': 0,
        'unknown_operators': [],
        'operator_details': [],
        'category_stats': {},
        'semantic_type_stats': {}
    }
    
    for operator in operators:
        op_info = get_operator_info(operator)
        validation_results['operator_details'].append(op_info)
        
        if op_info['exists']:
            validation_results['valid_operators'] += 1
            
            # 统计类别
            category = op_info['category']
            validation_results['category_stats'][category] = \
                validation_results['category_stats'].get(category, 0) + 1
            
            # 统计语义类型
            sem_type = op_info['semantic_type']
            validation_results['semantic_type_stats'][sem_type] = \
                validation_results['semantic_type_stats'].get(sem_type, 0) + 1
        else:
            validation_results['invalid_operators'] += 1
            validation_results['unknown_operators'].append(operator)
    
    validation_results['validation_rate'] = (
        validation_results['valid_operators'] / validation_results['total_operators']
        if validation_results['total_operators'] > 0 else 0
    )
    
    return validation_results