import re
from typing import List, Dict, Any, Optional, Union, Tuple
import logging

from .ast_nodes import (
    ASTNode, OperatorNode, VariableNode, ConstantNode, 
    UnaryMinusNode, ExpressionAST
)
from .operators import operator_registry, operator_semantics, get_operator_info

logger = logging.getLogger(__name__)

class Token:
    """
    词法单元
    """
    
    def __init__(self, type_: str, value: str, position: int):
        self.type = type_
        self.value = value
        self.position = position
    
    def __repr__(self):
        return f"Token({self.type}, '{self.value}', {self.position})"


class Lexer:
    """
    词法分析器
    
    将表达式字符串分解为词法单元
    """
    
    def __init__(self):
        # 词法规则（按优先级排序）
        self.token_patterns = [
            ('OPERATOR', r'ts_detrend|ts_avg|ts_norm|ts_std|ts_ret|ts_corr|ts_skew|ts_sub_mean|power_|mul_p|abs|div|mul|sub|rsi|sign'),
            ('VARIABLE', r'x\d+'),
            ('NUMBER', r'-?\d+\.?\d*'),
            ('LPAREN', r'\('),
            ('RPAREN', r'\)'),
            ('COMMA', r','),
            ('MINUS', r'-'),
            ('WHITESPACE', r'\s+'),
        ]
        
        # 编译正则表达式
        self.compiled_patterns = [(name, re.compile(pattern)) for name, pattern in self.token_patterns]
    
    def tokenize(self, text: str) -> List[Token]:
        """
        对输入文本进行词法分析
        
        Args:
            text: 输入表达式字符串
            
        Returns:
            词法单元列表
        """
        tokens = []
        position = 0
        
        while position < len(text):
            matched = False
            
            for token_type, pattern in self.compiled_patterns:
                match = pattern.match(text, position)
                if match:
                    value = match.group(0)
                    
                    # 跳过空白字符
                    if token_type != 'WHITESPACE':
                        tokens.append(Token(token_type, value, position))
                    
                    position = match.end()
                    matched = True
                    break
            
            if not matched:
                raise ValueError(f"无法识别的字符 '{text[position]}' at position {position}")
        
        return tokens


class Parser:
    """
    语法分析器
    
    使用递归下降法解析表达式，构建AST
    """
    
    def __init__(self):
        self.lexer = Lexer()
        self.tokens: List[Token] = []
        self.current_token_index = 0
        
    def parse(self, expression: str, variable_mapping: Dict[str, str]) -> ExpressionAST:
        """
        解析表达式
        
        Args:
            expression: 表达式字符串
            variable_mapping: 变量映射字典
            
        Returns:
            ExpressionAST对象
        """
        logger.info(f"开始解析表达式: {expression}")
        
        # 词法分析
        self.tokens = self.lexer.tokenize(expression)
        self.current_token_index = 0
        
        logger.debug(f"词法分析结果: {self.tokens}")
        
        # 语法分析
        root_node = self._parse_expression()
        
        # 检查是否有未处理的token
        if self.current_token_index < len(self.tokens):
            remaining_tokens = self.tokens[self.current_token_index:]
            logger.warning(f"表达式解析完成，但仍有未处理的token: {remaining_tokens}")
        
        # 创建AST
        ast = ExpressionAST(root_node, variable_mapping)
        
        logger.info(f"表达式解析成功: {ast.to_string()}")
        return ast
    
    def _current_token(self) -> Optional[Token]:
        """获取当前token"""
        if self.current_token_index < len(self.tokens):
            return self.tokens[self.current_token_index]
        return None
    
    def _consume_token(self, expected_type: str = None) -> Token:
        """
        消费当前token
        
        Args:
            expected_type: 期望的token类型
            
        Returns:
            消费的token
        """
        if self.current_token_index >= len(self.tokens):
            raise ValueError("意外的表达式结束")
        
        token = self.tokens[self.current_token_index]
        
        if expected_type and token.type != expected_type:
            raise ValueError(f"期望 {expected_type}，实际得到 {token.type}")
        
        self.current_token_index += 1
        return token
    
    def _parse_expression(self) -> ASTNode:
        """
        解析表达式（处理最外层的负号）
        
        Returns:
            AST节点
        """
        # 检查是否以负号开头
        current = self._current_token()
        if current and current.type == 'MINUS':
            self._consume_token('MINUS')
            
            # 创建一元负号节点
            minus_node = UnaryMinusNode()
            child_node = self._parse_primary()
            minus_node.add_child(child_node)
            
            return minus_node
        
        return self._parse_primary()
    
    def _parse_primary(self) -> ASTNode:
        """
        解析基本表达式
        
        Returns:
            AST节点
        """
        current = self._current_token()
        
        if not current:
            raise ValueError("意外的表达式结束")
        
        if current.type == 'OPERATOR':
            return self._parse_operator_call()
        elif current.type == 'VARIABLE':
            return self._parse_variable()
        elif current.type == 'NUMBER':
            return self._parse_number()
        elif current.type == 'LPAREN':
            # 处理括号表达式
            self._consume_token('LPAREN')
            node = self._parse_expression()
            self._consume_token('RPAREN')
            return node
        else:
            raise ValueError(f"意外的token类型: {current.type}")
    
    def _parse_operator_call(self) -> OperatorNode:
        """
        解析操作符调用
        
        Returns:
            操作符节点
        """
        # 获取操作符名称
        operator_token = self._consume_token('OPERATOR')
        operator_name = operator_token.value
        
        # 标准化操作符名称
        normalized_name = operator_semantics.normalize_operator(operator_name)
        
        # 获取操作符定义
        operator_def = operator_registry.get_operator(normalized_name)
        if operator_def:
            expected_arity = operator_def.arity
        else:
            logger.warning(f"未知操作符: {operator_name}")
            expected_arity = None
        
        # 创建操作符节点
        operator_node = OperatorNode(normalized_name, expected_arity)
        
        # 解析参数
        self._consume_token('LPAREN')
        
        # 解析第一个参数
        if self._current_token() and self._current_token().type != 'RPAREN':
            arg = self._parse_expression()
            operator_node.add_child(arg)
            
            # 解析其他参数
            while self._current_token() and self._current_token().type == 'COMMA':
                self._consume_token('COMMA')
                arg = self._parse_expression()
                operator_node.add_child(arg)
        
        self._consume_token('RPAREN')
        
        # 验证参数个数
        if expected_arity is not None and expected_arity >= 0:
            actual_arity = len(operator_node.children)
            if actual_arity != expected_arity:
                logger.warning(f"操作符 {operator_name} 期望 {expected_arity} 个参数，实际有 {actual_arity} 个")
        
        return operator_node
    
    def _parse_variable(self) -> VariableNode:
        """
        解析变量
        
        Returns:
            变量节点
        """
        variable_token = self._consume_token('VARIABLE')
        return VariableNode(variable_token.value)
    
    def _parse_number(self) -> ConstantNode:
        """
        解析数字常量
        
        Returns:
            常量节点
        """
        number_token = self._consume_token('NUMBER')
        
        # 转换为数字
        try:
            if '.' in number_token.value:
                value = float(number_token.value)
            else:
                value = int(number_token.value)
        except ValueError:
            raise ValueError(f"无效的数字: {number_token.value}")
        
        return ConstantNode(value)


class AlphaExpressionParser:
    """
    Alpha表达式解析器
    
    主要接口类，整合词法分析和语法分析
    """
    
    def __init__(self):
        self.parser = Parser()
        self._parse_statistics = {
            'total_parsed': 0,
            'successful_parses': 0,
            'failed_parses': 0,
            'parse_errors': []
        }
    
    def parse_expression(self, expression: str, variable_mapping: Dict[str, str]) -> ExpressionAST:
        """
        解析单个表达式
        
        Args:
            expression: 表达式字符串
            variable_mapping: 变量映射字典
            
        Returns:
            ExpressionAST对象
        """
        self._parse_statistics['total_parsed'] += 1
        
        try:
            ast = self.parser.parse(expression, variable_mapping)
            self._parse_statistics['successful_parses'] += 1
            return ast
        except Exception as e:
            self._parse_statistics['failed_parses'] += 1
            error_info = {
                'expression': expression,
                'error': str(e),
                'variable_mapping': variable_mapping
            }
            self._parse_statistics['parse_errors'].append(error_info)
            logger.error(f"表达式解析失败: {expression}, 错误: {str(e)}")
            raise
    
    def parse_multiple_expressions(self, expressions_data: List[Dict[str, Any]]) -> List[ExpressionAST]:
        """
        批量解析表达式
        
        Args:
            expressions_data: 表达式数据列表，每个元素包含expression和variable_mapping
            
        Returns:
            ExpressionAST对象列表
        """
        results = []
        
        for i, expr_data in enumerate(expressions_data):
            try:
                expression = expr_data['expression_info']['expression']
                variable_mapping = expr_data['expression_info']['variable_mapping']
                
                ast = self.parse_expression(expression, variable_mapping)
                results.append(ast)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"已解析 {i + 1} / {len(expressions_data)} 个表达式")
                    
            except Exception as e:
                logger.error(f"第 {i} 个表达式解析失败: {str(e)}")
                continue
        
        logger.info(f"批量解析完成: 成功 {len(results)} / {len(expressions_data)} 个表达式")
        return results
    
    def get_parse_statistics(self) -> Dict[str, Any]:
        """
        获取解析统计信息
        
        Returns:
            统计信息字典
        """
        stats = self._parse_statistics.copy()
        
        if stats['total_parsed'] > 0:
            stats['success_rate'] = stats['successful_parses'] / stats['total_parsed']
        else:
            stats['success_rate'] = 0.0
        
        return stats
    
    def validate_expression_syntax(self, expression: str) -> Dict[str, Any]:
        """
        验证表达式语法（不需要变量映射）
        
        Args:
            expression: 表达式字符串
            
        Returns:
            验证结果字典
        """
        try:
            # 使用空的变量映射进行语法验证
            ast = self.parse_expression(expression, {})
            
            # 运行AST验证
            validation_result = ast.validate()
            
            return {
                'syntax_valid': True,
                'ast_validation': validation_result,
                'ast_statistics': ast.get_statistics()
            }
            
        except Exception as e:
            return {
                'syntax_valid': False,
                'error': str(e),
                'ast_validation': None,
                'ast_statistics': None
            }
    
    def analyze_expression_complexity(self, expressions: List[str]) -> Dict[str, Any]:
        """
        分析表达式复杂度
        
        Args:
            expressions: 表达式字符串列表
            
        Returns:
            复杂度分析结果
        """
        complexities = []
        operator_usage = {}
        variable_usage = {}
        
        for expression in expressions:
            try:
                ast = self.parse_expression(expression, {})
                stats = ast.get_statistics()
                
                complexities.append({
                    'expression': expression,
                    'total_nodes': stats['total_nodes'],
                    'depth': stats['depth'],
                    'operator_count': stats['operator_count'],
                    'variable_count': stats['variable_count'],
                    'constant_count': stats['constant_count']
                })
                
                # 统计操作符使用
                for op in stats['operators']:
                    operator_usage[op] = operator_usage.get(op, 0) + 1
                
                # 统计变量使用
                for var in stats['variables']:
                    variable_usage[var] = variable_usage.get(var, 0) + 1
                    
            except Exception as e:
                logger.warning(f"复杂度分析失败: {expression}, 错误: {str(e)}")
                continue
        
        if not complexities:
            return {
                'total_expressions': len(expressions),
                'analyzed_expressions': 0,
                'error': '没有成功分析的表达式'
            }
        
        # 计算统计指标
        total_nodes = [c['total_nodes'] for c in complexities]
        depths = [c['depth'] for c in complexities]
        op_counts = [c['operator_count'] for c in complexities]
        
        return {
            'total_expressions': len(expressions),
            'analyzed_expressions': len(complexities),
            'complexity_stats': {
                'avg_nodes': sum(total_nodes) / len(total_nodes),
                'max_nodes': max(total_nodes),
                'min_nodes': min(total_nodes),
                'avg_depth': sum(depths) / len(depths),
                'max_depth': max(depths),
                'min_depth': min(depths),
                'avg_operators': sum(op_counts) / len(op_counts),
                'max_operators': max(op_counts),
                'min_operators': min(op_counts)
            },
            'operator_frequency': operator_usage,
            'variable_frequency': variable_usage,
            'parse_statistics': self.get_parse_statistics()
        }