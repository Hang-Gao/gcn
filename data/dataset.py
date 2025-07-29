import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class AlphaExpressionDataset(Dataset):
    """
    Alpha表达式数据集类，用于PyTorch训练
    
    这个数据集暂时只存储解析后的表达式信息和Sharpe值，
    图转换将在后续的graph_builder模块中实现
    """
    
    def __init__(self, parsed_data: List[Dict[str, Any]], transform=None):
        """
        初始化数据集
        
        Args:
            parsed_data: csv_parser解析后的数据列表
            transform: 可选的数据变换函数
        """
        self.data = parsed_data
        self.transform = transform
        
        # 提取Sharpe值用于快速访问
        self.sharpe_values = [item['sharpe'] for item in parsed_data]
        
        logger.info(f"初始化Alpha表达式数据集，样本数量: {len(self.data)}")
        
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取单个数据样本
        
        Args:
            idx: 样本索引
            
        Returns:
            包含表达式信息和目标值的字典
        """
        if idx >= len(self.data):
            raise IndexError(f"索引 {idx} 超出数据集范围 [0, {len(self.data)})")
        
        sample = {
            'expression_info': self.data[idx]['expression_info'],
            'sharpe': torch.tensor(self.data[idx]['sharpe'], dtype=torch.float32),
            'row_index': self.data[idx]['row_index']
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
    def get_sharpe_stats(self) -> Dict[str, float]:
        """
        获取Sharpe值统计信息
        
        Returns:
            包含min, max, mean, std的字典
        """
        sharpe_tensor = torch.tensor(self.sharpe_values, dtype=torch.float32)
        
        return {
            'min': sharpe_tensor.min().item(),
            'max': sharpe_tensor.max().item(), 
            'mean': sharpe_tensor.mean().item(),
            'std': sharpe_tensor.std().item()
        }
    
    def get_expression_by_idx(self, idx: int) -> str:
        """
        根据索引获取表达式字符串
        
        Args:
            idx: 样本索引
            
        Returns:
            表达式字符串
        """
        return self.data[idx]['expression_info']['expression']
    
    def get_variable_mapping_by_idx(self, idx: int) -> Dict[str, str]:
        """
        根据索引获取变量映射
        
        Args:
            idx: 样本索引
            
        Returns:
            变量映射字典 {x1: 'close', x2: 'open', ...}
        """
        return self.data[idx]['expression_info']['variable_mapping']
    
    def filter_by_sharpe_range(self, min_sharpe: Optional[float] = None, 
                              max_sharpe: Optional[float] = None) -> 'AlphaExpressionDataset':
        """
        根据Sharpe值范围过滤数据集
        
        Args:
            min_sharpe: 最小Sharpe值
            max_sharpe: 最大Sharpe值
            
        Returns:
            过滤后的新数据集
        """
        filtered_data = []
        
        for item in self.data:
            sharpe = item['sharpe']
            
            if min_sharpe is not None and sharpe < min_sharpe:
                continue
            if max_sharpe is not None and sharpe > max_sharpe:
                continue
                
            filtered_data.append(item)
        
        logger.info(f"根据Sharpe范围过滤: [{min_sharpe}, {max_sharpe}], "
                   f"原数量: {len(self.data)}, 过滤后: {len(filtered_data)}")
        
        return AlphaExpressionDataset(filtered_data, self.transform)
    
    def split_dataset(self, train_ratio: float = 0.8) -> tuple['AlphaExpressionDataset', 'AlphaExpressionDataset']:
        """
        将数据集分割为训练集和验证集
        
        Args:
            train_ratio: 训练集比例
            
        Returns:
            (train_dataset, val_dataset)
        """
        if not 0 < train_ratio < 1:
            raise ValueError(f"train_ratio必须在(0,1)范围内，当前值: {train_ratio}")
        
        total_size = len(self.data)
        train_size = int(total_size * train_ratio)
        
        # 简单的顺序分割，实际使用中可能需要随机分割
        train_data = self.data[:train_size]
        val_data = self.data[train_size:]
        
        logger.info(f"数据集分割: 总数{total_size}, 训练集{len(train_data)}, 验证集{len(val_data)}")
        
        return (AlphaExpressionDataset(train_data, self.transform),
                AlphaExpressionDataset(val_data, self.transform))
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        获取数据集详细信息
        
        Returns:
            数据集信息字典
        """
        # 统计变量使用情况
        variable_usage = {}
        expression_lengths = []
        
        for item in self.data:
            expr_info = item['expression_info']
            
            # 统计变量数量
            num_vars = expr_info.get('num_variables', 0)
            variable_usage[num_vars] = variable_usage.get(num_vars, 0) + 1
            
            # 统计表达式长度
            expr_length = len(expr_info.get('expression', ''))
            expression_lengths.append(expr_length)
        
        return {
            'total_samples': len(self.data),
            'sharpe_stats': self.get_sharpe_stats(),
            'variable_usage_distribution': variable_usage,
            'expression_length_stats': {
                'min': min(expression_lengths) if expression_lengths else 0,
                'max': max(expression_lengths) if expression_lengths else 0,
                'mean': sum(expression_lengths) / len(expression_lengths) if expression_lengths else 0
            }
        }


class AlphaDataLoader:
    """
    Alpha数据加载器，整合CSV解析和数据集创建
    """
    
    def __init__(self, csv_parser):
        """
        初始化数据加载器
        
        Args:
            csv_parser: CSVDataParser实例
        """
        self.csv_parser = csv_parser
        
    def load_train_dataset(self, train_csv_path: str, clean_data: bool = True) -> AlphaExpressionDataset:
        """
        加载训练数据集
        
        Args:
            train_csv_path: 训练CSV文件路径
            clean_data: 是否清洗数据
            
        Returns:
            AlphaExpressionDataset实例
        """
        logger.info(f"加载训练数据集: {train_csv_path}")
        
        # 解析CSV数据
        parsed_data = self.csv_parser.parse_dataset(train_csv_path)
        
        # 可选的数据清洗
        if clean_data:
            parsed_data = self.csv_parser.clean_data(parsed_data)
        
        # 创建数据集
        dataset = AlphaExpressionDataset(parsed_data)
        
        logger.info(f"训练数据集加载完成，样本数量: {len(dataset)}")
        return dataset
    
    def load_test_dataset(self, test_csv_path: str, clean_data: bool = True) -> AlphaExpressionDataset:
        """
        加载测试数据集
        
        Args:
            test_csv_path: 测试CSV文件路径
            clean_data: 是否清洗数据
            
        Returns:
            AlphaExpressionDataset实例
        """
        logger.info(f"加载测试数据集: {test_csv_path}")
        
        # 解析CSV数据
        parsed_data = self.csv_parser.parse_dataset(test_csv_path)
        
        # 可选的数据清洗
        if clean_data:
            parsed_data = self.csv_parser.clean_data(parsed_data)
        
        # 创建数据集
        dataset = AlphaExpressionDataset(parsed_data)
        
        logger.info(f"测试数据集加载完成，样本数量: {len(dataset)}")
        return dataset
    
    def load_both_datasets(self, train_csv_path: str, test_csv_path: str, 
                          clean_data: bool = True) -> tuple[AlphaExpressionDataset, AlphaExpressionDataset]:
        """
        同时加载训练和测试数据集
        
        Args:
            train_csv_path: 训练CSV文件路径
            test_csv_path: 测试CSV文件路径
            clean_data: 是否清洗数据
            
        Returns:
            (train_dataset, test_dataset)
        """
        train_dataset = self.load_train_dataset(train_csv_path, clean_data)
        test_dataset = self.load_test_dataset(test_csv_path, clean_data)
        
        return train_dataset, test_dataset


class AlphaDataset(Dataset):
    """
    用于GCN的Alpha数据集，包含图转换功能
    """
    
    def __init__(self, data_list: List[Dict[str, Any]], graph_builder, transform_graphs: bool = True):
        """
        初始化数据集
        
        Args:
            data_list: 包含表达式信息和AST的数据列表
            graph_builder: 图构建器实例
            transform_graphs: 是否立即转换为图数据
        """
        self.data_list = data_list
        self.graph_builder = graph_builder
        self.transform_graphs = transform_graphs
        
        if transform_graphs:
            self.graphs = self._build_all_graphs()
        else:
            self.graphs = None
        
        logger.info(f"AlphaDataset初始化完成，样本数量: {len(self.data_list)}")
    
    def _build_all_graphs(self) -> List[Data]:
        """构建所有图数据"""
        graphs = []
        
        for i, data_item in enumerate(self.data_list):
            try:
                if 'ast' in data_item and data_item['ast'] is not None:
                    graph = self.graph_builder.ast_to_graph(
                        data_item['ast'], 
                        data_item['sharpe']
                    )
                    graphs.append(graph)
                else:
                    logger.warning(f"样本 {i} 缺少有效的AST")
                    
            except Exception as e:
                logger.error(f"样本 {i} 图转换失败: {str(e)}")
                continue
        
        logger.info(f"图转换完成: {len(graphs)} / {len(self.data_list)}")
        return graphs
    
    def __len__(self) -> int:
        if self.transform_graphs:
            return len(self.graphs) if self.graphs else 0
        else:
            return len(self.data_list)
    
    def __getitem__(self, idx: int) -> Data:
        if self.transform_graphs:
            if not self.graphs or idx >= len(self.graphs):
                raise IndexError(f"索引 {idx} 超出范围")
            return self.graphs[idx]
        else:
            # 动态转换
            data_item = self.data_list[idx]
            if 'ast' not in data_item:
                raise ValueError(f"样本 {idx} 缺少AST")
            
            return self.graph_builder.ast_to_graph(
                data_item['ast'], 
                data_item['sharpe']
            )
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """获取图统计信息"""
        if not self.transform_graphs or not self.graphs:
            return {}
        
        num_nodes = [g.num_nodes for g in self.graphs]
        num_edges = [g.edge_index.size(1) for g in self.graphs]
        targets = [g.y.item() for g in self.graphs if g.y is not None]
        
        return {
            'num_graphs': len(self.graphs),
            'node_stats': {
                'min': min(num_nodes) if num_nodes else 0,
                'max': max(num_nodes) if num_nodes else 0,
                'mean': sum(num_nodes) / len(num_nodes) if num_nodes else 0
            },
            'edge_stats': {
                'min': min(num_edges) if num_edges else 0,
                'max': max(num_edges) if num_edges else 0,
                'mean': sum(num_edges) / len(num_edges) if num_edges else 0
            },
            'target_stats': {
                'min': min(targets) if targets else 0,
                'max': max(targets) if targets else 0,
                'mean': sum(targets) / len(targets) if targets else 0
            } if targets else {}
        }