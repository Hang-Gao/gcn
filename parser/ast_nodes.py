from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

class ASTNode(ABC):
    """
    抽象语法树节点基类
    
    所有表达式节点都继承自这个基类
    """
    
    def __init__(self, node_type: str):
        self.node_type = node_type
        self.children: List['ASTNode'] = []
        self.parent: Optional['ASTNode'] = None
        
    @abstractmethod
    def evaluate(self, variable_values: Dict[str, Any]) -> Any:
        """
        计算节点值（用于将来的数值计算，当前阶段主要用于结构验证）
        
        Args:
            variable_values: 变量值字典
            
        Returns:
            计算结果
        """
        pass
    
    @abstractmethod
    def to_string(self) -> str:
        """
        将节点转换为字符串表示
        
        Returns:
            字符串表示
        """
        pass
    
    def add_child(self, child: 'ASTNode') -> None:
        """添加子节点"""
        self.children.append(child)
        child.parent = self
    
    def get_depth(self) -> int:
        """获取节点深度"""
        if not self.children:
            return 1
        return 1 + max(child.get_depth() for child in self.children)
    
    def get_node_count(self) -> int:
        """获取子树中节点总数"""
        count = 1
        for child in self.children:
            count += child.get_node_count()
        return count
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将AST节点转换为字典表示（用于图神经网络）
        
        Returns:
            节点信息字典
        """
        return {
            'type': self.node_type,
            'children': [child.to_dict() for child in self.children],
            'depth': self.get_depth(),
            'node_count': self.get_node_count()
        }


class OperatorNode(ASTNode):
    """
    操作符节点
    
    表示各种运算操作，如ts_detrend, abs, div等
    """
    
    def __init__(self, operator: str, arity: int = None):
        super().__init__('operator')
        self.operator = operator
        self.arity = arity  # 操作符的参数个数
        
    def evaluate(self, variable_values: Dict[str, Any]) -> Any:
        """
        计算操作符的值
        """
        # 当前阶段返回占位符，实际计算在图神经网络中完成
        child_values = [child.evaluate(variable_values) for child in self.children]
        return f"{self.operator}({', '.join(map(str, child_values))})"
    
    def to_string(self) -> str:
        """
        操作符的字符串表示
        """
        if not self.children:
            return self.operator
        
        child_strs = [child.to_string() for child in self.children]
        return f"{self.operator}({', '.join(child_strs)})"
    
    def validate_arity(self) -> bool:
        """
        验证操作符参数个数是否正确
        
        Returns:
            是否有效
        """
        if self.arity is None:
            return True  # 不限制参数个数
        return len(self.children) == self.arity
    
    def to_dict(self) -> Dict[str, Any]:
        """
        操作符节点的字典表示
        """
        base_dict = super().to_dict()
        base_dict.update({
            'operator': self.operator,
            'arity': self.arity,
            'child_count': len(self.children)
        })
        return base_dict


class VariableNode(ASTNode):
    """
    变量节点
    
    表示x1, x2, x3等变量
    """
    
    def __init__(self, variable_name: str):
        super().__init__('variable')
        self.variable_name = variable_name
        
    def evaluate(self, variable_values: Dict[str, Any]) -> Any:
        """
        获取变量的值
        """
        return variable_values.get(self.variable_name, f"${self.variable_name}")
    
    def to_string(self) -> str:
        """
        变量的字符串表示
        """
        return self.variable_name
    
    def to_dict(self) -> Dict[str, Any]:
        """
        变量节点的字典表示
        """
        base_dict = super().to_dict()
        base_dict.update({
            'variable_name': self.variable_name
        })
        return base_dict


class ConstantNode(ASTNode):
    """
    常量节点
    
    表示数值常量，如1110, 225等
    """
    
    def __init__(self, value: Union[int, float]):
        super().__init__('constant')
        self.value = value
        
    def evaluate(self, variable_values: Dict[str, Any]) -> Any:
        """
        返回常量值
        """
        return self.value
    
    def to_string(self) -> str:
        """
        常量的字符串表示
        """
        return str(self.value)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        常量节点的字典表示
        """
        base_dict = super().to_dict()
        base_dict.update({
            'value': self.value,
            'value_type': type(self.value).__name__
        })
        return base_dict


class UnaryMinusNode(ASTNode):
    """
    一元负号节点
    
    特殊处理负号，如-sub(...)
    """
    
    def __init__(self):
        super().__init__('unary_minus')
        
    def evaluate(self, variable_values: Dict[str, Any]) -> Any:
        """
        计算负值
        """
        if len(self.children) != 1:
            raise ValueError("一元负号必须只有一个子节点")
        
        child_value = self.children[0].evaluate(variable_values)
        return f"-({child_value})"
    
    def to_string(self) -> str:
        """
        负号的字符串表示
        """
        if len(self.children) != 1:
            return "-"
        
        return f"-{self.children[0].to_string()}"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        一元负号节点的字典表示
        """
        return super().to_dict()


class ExpressionAST:
    """
    完整的表达式抽象语法树
    
    包含根节点和相关的元数据
    """
    
    def __init__(self, root: ASTNode, variable_mapping: Dict[str, str]):
        self.root = root
        self.variable_mapping = variable_mapping  # x1 -> 'close', x2 -> 'open' 等
        
    def evaluate(self, feature_values: Dict[str, Any] = None) -> Any:
        """
        计算整个表达式的值
        
        Args:
            feature_values: 特征值字典，如 {'close': 100, 'open': 99}
            
        Returns:
            表达式计算结果
        """
        if feature_values is None:
            feature_values = {}
        
        # 将特征值映射到变量
        variable_values = {}
        for var, feature in self.variable_mapping.items():
            variable_values[var] = feature_values.get(feature, f"${feature}")
        
        return self.root.evaluate(variable_values)
    
    def to_string(self) -> str:
        """
        将整个AST转换为字符串
        """
        return self.root.to_string()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取AST统计信息
        
        Returns:
            统计信息字典
        """
        def collect_node_types(node: ASTNode) -> Dict[str, int]:
            """递归收集节点类型统计"""
            stats = {node.node_type: 1}
            
            for child in node.children:
                child_stats = collect_node_types(child)
                for node_type, count in child_stats.items():
                    stats[node_type] = stats.get(node_type, 0) + count
            
            return stats
        
        def collect_operators(node: ASTNode) -> List[str]:
            """递归收集所有操作符"""
            operators = []
            
            if isinstance(node, OperatorNode):
                operators.append(node.operator)
            
            for child in node.children:
                operators.extend(collect_operators(child))
            
            return operators
        
        def collect_variables(node: ASTNode) -> List[str]:
            """递归收集所有变量"""
            variables = []
            
            if isinstance(node, VariableNode):
                variables.append(node.variable_name)
            
            for child in node.children:
                variables.extend(collect_variables(child))
            
            return variables
        
        def collect_constants(node: ASTNode) -> List[Union[int, float]]:
            """递归收集所有常量"""
            constants = []
            
            if isinstance(node, ConstantNode):
                constants.append(node.value)
            
            for child in node.children:
                constants.extend(collect_constants(child))
            
            return constants
        
        node_type_stats = collect_node_types(self.root)
        operators = collect_operators(self.root)
        variables = collect_variables(self.root)
        constants = collect_constants(self.root)
        
        return {
            'total_nodes': self.root.get_node_count(),
            'depth': self.root.get_depth(),
            'node_type_distribution': node_type_stats,
            'operators': operators,
            'operator_count': len(operators),
            'unique_operators': list(set(operators)),
            'variables': variables,
            'variable_count': len(variables),
            'unique_variables': list(set(variables)),
            'constants': constants,
            'constant_count': len(constants),
            'variable_mapping': self.variable_mapping
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将整个AST转换为字典表示
        
        Returns:
            AST字典表示
        """
        return {
            'root': self.root.to_dict(),
            'variable_mapping': self.variable_mapping,
            'statistics': self.get_statistics()
        }
    
    def validate(self) -> Dict[str, Any]:
        """
        验证AST的正确性
        
        Returns:
            验证结果字典
        """
        validation_errors = []
        validation_warnings = []
        
        def validate_node(node: ASTNode) -> None:
            """递归验证节点"""
            if isinstance(node, OperatorNode):
                # 验证操作符参数个数
                if not node.validate_arity():
                    validation_errors.append(
                        f"操作符 {node.operator} 期望 {node.arity} 个参数，实际有 {len(node.children)} 个"
                    )
                
                # 检查未知操作符（基于已知的16个操作符）
                known_operators = [
                    'ts_detrend', 'ts_avg', 'abs', 'div', 'mul', 'mul_p', 'power_',
                    'rsi', 'sign', 'sub', 'ts_corr', 'ts_norm', 'ts_ret', 'ts_skew',
                    'ts_std', 'ts_sub_mean'
                ]
                
                if node.operator not in known_operators:
                    validation_warnings.append(f"未知操作符: {node.operator}")
            
            elif isinstance(node, VariableNode):
                # 检查变量是否在映射中
                if node.variable_name not in self.variable_mapping:
                    validation_warnings.append(f"变量 {node.variable_name} 未在变量映射中找到")
            
            elif isinstance(node, UnaryMinusNode):
                # 验证一元负号只有一个子节点
                if len(node.children) != 1:
                    validation_errors.append("一元负号必须只有一个子节点")
            
            # 递归验证子节点
            for child in node.children:
                validate_node(child)
        
        validate_node(self.root)
        
        return {
            'is_valid': len(validation_errors) == 0,
            'errors': validation_errors,
            'warnings': validation_warnings,
            'error_count': len(validation_errors),
            'warning_count': len(validation_warnings)
        }