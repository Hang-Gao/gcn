import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from typing import List, Dict, Any, Tuple, Optional, Union
import numpy as np
import logging

from parser.ast_nodes import (
    ASTNode, OperatorNode, VariableNode, ConstantNode, 
    UnaryMinusNode, ExpressionAST
)
from parser.operators import operator_registry

logger = logging.getLogger(__name__)

class NodeFeatureEncoder:
    """
    节点特征编码器
    
    将AST节点转换为固定维度的特征向量
    """
    
    def __init__(self):
        # 获取所有已知操作符
        self.operators = operator_registry.get_operator_names()
        self.operator_to_idx = {op: i for i, op in enumerate(self.operators)}
        
        # 节点类型映射
        self.node_types = ['operator', 'variable', 'constant', 'unary_minus']
        self.node_type_to_idx = {nt: i for i, nt in enumerate(self.node_types)}
        
        # 特征名称映射（动态变量x1,x2,x3可以映射到这些特征）
        self.feature_names = ['Ret', 'open', 'high', 'low', 'close', 'vol', 'oi']
        self.feature_to_idx = {fname: i for i, fname in enumerate(self.feature_names)}
        
        # 特征维度计算
        self.operator_dim = len(self.operators)  # 操作符one-hot
        self.node_type_dim = len(self.node_types)  # 节点类型one-hot
        self.feature_dim_size = len(self.feature_names)  # 特征类型one-hot（替代简单的变量编码）
        self.constant_dim = 1   # 常量值（标准化后）
        self.meta_dim = 3       # 元数据：子节点数、深度、位置编码
        
        # 总特征维度
        self.feature_dim = (
            self.operator_dim + self.node_type_dim + 
            self.feature_dim_size + self.constant_dim + self.meta_dim
        )
        
        logger.info(f"节点特征编码器初始化完成，特征维度: {self.feature_dim}")
        logger.info(f"  操作符维度: {self.operator_dim}")
        logger.info(f"  节点类型维度: {self.node_type_dim}")
        logger.info(f"  特征名称维度: {self.feature_dim_size}")
        logger.info(f"  常量维度: {self.constant_dim}")
        logger.info(f"  元数据维度: {self.meta_dim}")
    
    def encode_node(self, node: ASTNode, depth: int = 0, position: int = 0, 
                   variable_mapping: Dict[str, str] = None) -> torch.Tensor:
        """
        编码单个节点
        
        Args:
            node: AST节点
            depth: 节点在树中的深度
            position: 节点在同层中的位置
            variable_mapping: 变量映射字典 {x1: 'close', x2: 'open', ...}
            
        Returns:
            特征向量 (feature_dim,)
        """
        # 初始化特征向量
        features = torch.zeros(self.feature_dim)
        offset = 0
        
        # 1. 操作符特征 (one-hot)
        operator_features = torch.zeros(self.operator_dim)
        if isinstance(node, OperatorNode):
            if node.operator in self.operator_to_idx:
                operator_features[self.operator_to_idx[node.operator]] = 1.0
        features[offset:offset + self.operator_dim] = operator_features
        offset += self.operator_dim
        
        # 2. 节点类型特征 (one-hot)
        node_type_features = torch.zeros(self.node_type_dim)
        node_type = node.node_type
        
        # 处理UnaryMinusNode的特殊情况
        from parser.ast_nodes import UnaryMinusNode
        if isinstance(node, UnaryMinusNode):
            node_type = 'unary_minus'
        
        if node_type in self.node_type_to_idx:
            node_type_features[self.node_type_to_idx[node_type]] = 1.0
        features[offset:offset + self.node_type_dim] = node_type_features
        offset += self.node_type_dim
        
        # 3. 变量特征 (使用动态映射到具体特征名称)
        feature_name_features = torch.zeros(self.feature_dim_size)
        if isinstance(node, VariableNode):
            var_name = node.variable_name  # 如 'x1', 'x2', 'x3'
            
            # 使用variable_mapping将x变量映射到具体特征名称
            if variable_mapping and var_name in variable_mapping:
                feature_name = variable_mapping[var_name]  # 如 'close', 'open' 等
                if feature_name in self.feature_to_idx:
                    feature_idx = self.feature_to_idx[feature_name]
                    feature_name_features[feature_idx] = 1.0
                else:
                    logger.warning(f"未知的特征名称: {feature_name}")
            else:
                logger.warning(f"变量 {var_name} 没有在variable_mapping中找到映射")
                
        features[offset:offset + self.feature_dim_size] = feature_name_features
        offset += self.feature_dim_size
        
        # 4. 常量特征 (标准化数值)
        constant_features = torch.zeros(self.constant_dim)
        if isinstance(node, ConstantNode):
            # 对常量进行标准化处理
            normalized_value = self._normalize_constant(node.value)
            constant_features[0] = normalized_value
        features[offset:offset + self.constant_dim] = constant_features
        offset += self.constant_dim
        
        # 5. 元数据特征
        meta_features = torch.tensor([
            len(node.children) / 5.0,  # 子节点数标准化（最大假设5个子节点）
            depth / 10.0,              # 深度标准化（最大假设10层）
            position / 20.0            # 位置标准化（最大假设20个同层节点）
        ], dtype=torch.float32)
        features[offset:offset + self.meta_dim] = meta_features
        
        return features
    
    def _normalize_constant(self, value: Union[int, float]) -> float:
        """
        标准化常量值
        
        Args:
            value: 常量值
            
        Returns:
            标准化后的值 [-1, 1]
        """
        # 使用tanh函数将值映射到[-1, 1]区间
        # 对于很大的数值进行缩放
        if abs(value) > 1000:
            value = value / 1000.0
        
        return float(np.tanh(value / 100.0))


class EdgeBuilder:
    """
    边构建器
    
    根据AST结构构建图的边连接
    """
    
    def __init__(self):
        self.edge_types = ['parent_child', 'child_parent']  # 双向边
        
    def build_edges(self, node_list: List[ASTNode]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        构建边索引和边特征
        
        Args:
            node_list: 按顺序排列的节点列表
            
        Returns:
            (edge_index, edge_attr)
            edge_index: (2, num_edges) 边索引
            edge_attr: (num_edges, edge_feature_dim) 边特征
        """
        node_to_idx = {id(node): idx for idx, node in enumerate(node_list)}
        edges = []
        
        # 遍历所有节点，建立父子关系的边
        for node in node_list:
            parent_idx = node_to_idx[id(node)]
            
            for child in node.children:
                if id(child) in node_to_idx:
                    child_idx = node_to_idx[id(child)]
                    
                    # 添加父到子的边
                    edges.append([parent_idx, child_idx])
                    # 添加子到父的边（无向图）
                    edges.append([child_idx, parent_idx])
        
        if not edges:
            # 处理单节点图的情况
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 2), dtype=torch.float)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            
            # 简单的边特征：[是否为父到子, 是否为子到父]
            num_edges = len(edges)
            edge_attr = torch.zeros((num_edges, 2))
            
            # 奇数索引是父到子，偶数索引是子到父
            for i in range(0, num_edges, 2):
                if i < num_edges:
                    edge_attr[i, 0] = 1.0      # 父到子
                if i + 1 < num_edges:
                    edge_attr[i + 1, 1] = 1.0  # 子到父
        
        return edge_index, edge_attr


class GraphBuilder:
    """
    图构建器
    
    将表达式AST转换为PyTorch Geometric图数据
    """
    
    def __init__(self):
        self.node_encoder = NodeFeatureEncoder()
        self.edge_builder = EdgeBuilder()
        self._vocabulary_built = False
        
    def ast_to_graph(self, ast: ExpressionAST, target_value: Optional[float] = None) -> Data:
        """
        将AST转换为图数据
        
        Args:
            ast: 表达式AST
            target_value: 目标值（如Sharpe值）
            
        Returns:
            PyTorch Geometric Data对象
        """
        # 1. 收集所有节点（深度优先遍历）
        # 修改：收集节点的同时记录深度
        node_list_with_depth = self._collect_nodes_with_depth(ast.root)
        node_list = [item[0] for item in node_list_with_depth]
        
        # 2. 编码节点特征
        node_features = []
        for i, (node, depth) in enumerate(node_list_with_depth):
            position = i  # 简单使用遍历顺序作为位置
            features = self.node_encoder.encode_node(node, depth, position, ast.variable_mapping)
            node_features.append(features)
        
        # 3. 构建边
        edge_index, edge_attr = self.edge_builder.build_edges(node_list)
        
        # 4. 创建图数据
        x = torch.stack(node_features)  # (num_nodes, feature_dim)
        
        # 5. 添加目标值
        y = torch.tensor([target_value], dtype=torch.float32) if target_value is not None else None
        
        # 6. 添加图级别的特征
        graph_features = self._extract_graph_features(ast, len(node_list))
        
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            graph_features=graph_features,
            num_nodes=len(node_list)
        )
        
        # 不添加variable_mapping等属性到Data对象中，避免batch化时的KeyError
        # 这些信息可以通过其他方式管理
        
        return data
    
    def _collect_nodes_with_depth(self, root: ASTNode) -> List[Tuple[ASTNode, int]]:
        """
        深度优先收集所有节点及其深度
        
        Args:
            root: 根节点
            
        Returns:
            (节点, 深度)元组的列表
        """
        nodes = []
        
        def dfs(node, depth):
            nodes.append((node, depth))
            for child in node.children:
                dfs(child, depth + 1)
        
        dfs(root, 0)
        return nodes
    
    def _extract_graph_features(self, ast: ExpressionAST, num_nodes: int) -> torch.Tensor:
        """
        提取图级别特征
        
        Args:
            ast: 表达式AST
            num_nodes: 节点总数
            
        Returns:
            图特征向量
        """
        stats = ast.get_statistics()
        
        # 图级别特征
        graph_features = torch.tensor([
            num_nodes / 30.0,                    # 节点数标准化
            stats['depth'] / 10.0,               # 深度标准化
            stats['operator_count'] / 15.0,      # 操作符数标准化
            stats['variable_count'] / 10.0,      # 变量数标准化
            stats['constant_count'] / 10.0,      # 常量数标准化
            len(stats['unique_operators']) / 16.0,  # 唯一操作符数标准化
            len(stats['unique_variables']) / 10.0   # 唯一变量数标准化
        ], dtype=torch.float32)
        
        return graph_features
    
    def batch_ast_to_graphs(self, asts: List[ExpressionAST], 
                           target_values: Optional[List[float]] = None) -> List[Data]:
        """
        批量转换AST为图数据
        
        Args:
            asts: AST列表
            target_values: 目标值列表
            
        Returns:
            图数据列表
        """
        graphs = []
        
        for i, ast in enumerate(asts):
            target = target_values[i] if target_values else None
            
            try:
                graph = self.ast_to_graph(ast, target)
                graphs.append(graph)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"已转换 {i + 1} / {len(asts)} 个AST为图")
                    
            except Exception as e:
                logger.error(f"AST {i} 转换失败: {str(e)}")
                continue
        
        logger.info(f"批量图转换完成: 成功 {len(graphs)} / {len(asts)} 个")
        return graphs
    
    def build_vocabulary(self, asts: List[ExpressionAST]) -> None:
        """
        构建词汇表（实际上NodeFeatureEncoder已经有固定的词汇表）
        这个方法主要用于兼容性，确保接口一致
        
        Args:
            asts: AST列表
        """
        self._vocabulary_built = True
        logger.info(f"词汇表构建完成，基于 {len(asts)} 个AST")
    
    def get_vocabulary_size(self) -> int:
        """
        获取词汇表大小
        
        Returns:
            词汇表大小
        """
        return self.node_encoder.feature_dim
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """
        获取特征维度信息
        
        Returns:
            特征维度字典
        """
        # 需要检查一个实际的图对象来获取真实的graph_features维度
        graph_feature_dim = 7  # 默认值
        
        return {
            'node_feature_dim': self.node_encoder.feature_dim,
            'edge_feature_dim': 2,
            'graph_feature_dim': graph_feature_dim,
            'operator_vocab_size': len(self.node_encoder.operators),
            'node_type_vocab_size': len(self.node_encoder.node_types),
            'feature_name_vocab_size': len(self.node_encoder.feature_names)
        }


class GraphDataset:
    """
    图数据集
    
    管理批量图数据，支持PyTorch Geometric DataLoader
    """
    
    def __init__(self, graphs: List[Data]):
        self.graphs = graphs
        
    def __len__(self) -> int:
        return len(self.graphs)
    
    def __getitem__(self, idx: int) -> Data:
        return self.graphs[idx]
    
    def get_batch(self, indices: List[int]) -> Batch:
        """
        获取批次数据
        
        Args:
            indices: 索引列表
            
        Returns:
            批次图数据
        """
        batch_graphs = [self.graphs[i] for i in indices]
        return Batch.from_data_list(batch_graphs)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取数据集统计信息
        
        Returns:
            统计信息字典
        """
        if not self.graphs:
            return {}
        
        num_nodes = [g.num_nodes for g in self.graphs]
        num_edges = [g.edge_index.size(1) for g in self.graphs]
        
        # 目标值统计（如果存在）
        targets = []
        for g in self.graphs:
            if g.y is not None:
                targets.append(g.y.item())
        
        stats = {
            'num_graphs': len(self.graphs),
            'node_stats': {
                'min': min(num_nodes),
                'max': max(num_nodes),
                'mean': sum(num_nodes) / len(num_nodes),
                'total': sum(num_nodes)
            },
            'edge_stats': {
                'min': min(num_edges),
                'max': max(num_edges),
                'mean': sum(num_edges) / len(num_edges),
                'total': sum(num_edges)
            }
        }
        
        if targets:
            stats['target_stats'] = {
                'min': min(targets),
                'max': max(targets),
                'mean': sum(targets) / len(targets),
                'count': len(targets)
            }
        
        return stats


def create_graph_dataset_from_expressions(expression_data: List[Dict[str, Any]], 
                                        parser, 
                                        include_targets: bool = True) -> GraphDataset:
    """
    从表达式数据创建图数据集
    
    Args:
        expression_data: 表达式数据列表
        parser: Alpha表达式解析器
        include_targets: 是否包含目标值
        
    Returns:
        图数据集
    """
    logger.info(f"开始从 {len(expression_data)} 个表达式创建图数据集")
    
    # 1. 解析所有表达式为AST
    asts = []
    targets = []
    
    for data in expression_data:
        try:
            expr_info = data['expression_info']
            ast = parser.parse_expression(
                expr_info['expression'],
                expr_info['variable_mapping']
            )
            asts.append(ast)
            
            if include_targets:
                targets.append(data['sharpe'])
                
        except Exception as e:
            logger.warning(f"表达式解析失败，跳过: {str(e)}")
            continue
    
    # 2. 转换AST为图
    graph_builder = GraphBuilder()
    target_list = targets if include_targets else None
    graphs = graph_builder.batch_ast_to_graphs(asts, target_list)
    
    # 3. 创建数据集
    dataset = GraphDataset(graphs)
    
    logger.info(f"图数据集创建完成: {len(dataset)} 个图")
    return dataset