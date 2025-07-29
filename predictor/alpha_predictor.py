import sys
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import logging

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model.gcn_model import AlphaGCN, create_alpha_gcn_model
from model.graph_builder import GraphBuilder
from parser.expression_parser import AlphaExpressionParser
from data.csv_parser import CSVDataParser
from predictor.model_manager import ModelManager

logger = logging.getLogger(__name__)


class AlphaPredictor:
    """
    Alpha表达式预测器
    
    独立的预测系统，加载训练好的模型进行Sharpe值预测
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: Optional[str] = None):
        """
        初始化预测器
        
        Args:
            model_path: 模型文件路径，如果为None则自动选择最新模型
            device: 计算设备 ('cpu', 'cuda', 'mps' 或 None自动选择)
        """
        self.device = self._setup_device(device)
        self.model_manager = ModelManager()
        
        # 选择和加载模型
        self.model_path = self._select_model(model_path)
        self.model = None
        self.feature_dims = None
        
        # 初始化解析器和图构建器
        self.expression_parser = AlphaExpressionParser()
        self.csv_parser = CSVDataParser()
        self.graph_builder = None
        
        # 加载模型
        self._load_model()
        
        logger.info(f"AlphaPredictor初始化完成")
        logger.info(f"  使用模型: {self.model_path}")
        logger.info(f"  计算设备: {self.device}")
    
    def _setup_device(self, device: Optional[str]) -> torch.device:
        """设置计算设备"""
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        return torch.device(device)
    
    def _select_model(self, model_path: Optional[str]) -> str:
        """选择模型文件"""
        selected_path = self.model_manager.select_model(model_path)
        if selected_path is None:
            raise FileNotFoundError("未找到任何可用的模型文件")
        
        return selected_path
    
    def _load_model(self):
        """加载训练好的模型"""
        try:
            # 加载模型检查点
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # 提取模型配置
            if 'config' in checkpoint:
                config = checkpoint['config']
                model_config = {
                    'hidden_dim': config['model']['hidden_dim'],
                    'num_layers': config['model']['num_layers'],
                    'dropout': config['model']['dropout'],
                    'pool_method': 'mean',
                    'use_edge_features': True,
                    'use_graph_features': True,
                    'use_attention': False
                }
                # 从检查点提取vocab_size作为节点特征维度
                vocab_size = checkpoint.get('vocab_size', 31)  # 默认值
                self.feature_dims = {
                    'node_feature_dim': vocab_size,
                    'edge_feature_dim': 2,
                    'graph_feature_dim': 7
                }
            else:
                # 如果没有配置信息，使用默认配置
                logger.warning("模型文件中未找到配置信息，使用默认配置")
                model_config = {
                    'hidden_dim': 128,
                    'num_layers': 3,
                    'dropout': 0.2,
                    'pool_method': 'mean',
                    'use_edge_features': True,
                    'use_graph_features': True,
                    'use_attention': False
                }
                # 使用默认特征维度
                self.feature_dims = {
                    'node_feature_dim': 50,  # 需要根据实际情况调整
                    'edge_feature_dim': 2,
                    'graph_feature_dim': 7
                }
            
            # 创建模型
            self.model = create_alpha_gcn_model(self.feature_dims, model_config)
            
            # 加载模型权重
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # 兼容旧版本的模型文件
                self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            
            # 初始化图构建器
            self.graph_builder = GraphBuilder()
            
            logger.info(f"模型加载成功: {Path(self.model_path).name}")
            
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {e}")
    
    def parse_expression(self, expression_str: str) -> Dict[str, Any]:
        """
        解析alpha表达式字符串
        
        Args:
            expression_str: alpha表达式字符串，支持两种格式：
                1. 简化格式: "表达式;num 数据1名称 数据1编号 数据2名称 数据2编号..."
                2. 完整格式: "name::表达式;num 数据编号::其他参数"
                
        Returns:
            解析结果字典
        """
        try:
            # 检查是否是完整的CSV格式（包含::分隔符）
            if "::" in expression_str:
                # 完整格式，直接使用CSV解析器
                result = self.csv_parser.parse_expression_string(expression_str)
            else:
                # 简化格式，需要构造为完整格式再解析
                # 添加临时的name前缀
                full_format = f"temp_alpha::{expression_str}::dummy"
                result = self.csv_parser.parse_expression_string(full_format)
            
            if result is None:
                raise ValueError(f"表达式解析失败: {expression_str}")
            
            return result
            
        except Exception as e:
            raise ValueError(f"表达式解析错误: {e}")
    
    def predict_single(self, expression_str: str) -> Dict[str, Any]:
        """
        预测单个alpha表达式的Sharpe值
        
        Args:
            expression_str: alpha表达式字符串
            
        Returns:
            预测结果字典
        """
        try:
            # 解析表达式
            parsed_result = self.parse_expression(expression_str)
            
            # 使用表达式解析器构建AST
            ast = self.expression_parser.parse_expression(
                parsed_result['expression'], 
                parsed_result['variable_mapping']
            )
            
            # 构建图
            graph_data = self.graph_builder.ast_to_graph(ast)
            
            # 转换为PyTorch Geometric格式
            from torch_geometric.data import Batch
            batch = Batch.from_data_list([graph_data])
            batch = batch.to(self.device)
            
            # 预测
            with torch.no_grad():
                prediction = self.model(batch)
                sharpe_value = prediction.cpu().numpy()[0, 0]
            
            return {
                'expression': expression_str,
                'predicted_sharpe': float(sharpe_value),
                'parsed_expression': parsed_result['expression'],
                'variable_mapping': parsed_result['variable_mapping']
            }
            
        except Exception as e:
            return {
                'expression': expression_str,
                'error': str(e),
                'predicted_sharpe': None
            }
    
    def predict_batch(self, expressions: List[str]) -> List[Dict[str, Any]]:
        """
        批量预测多个alpha表达式
        
        Args:
            expressions: alpha表达式列表
            
        Returns:
            预测结果列表
        """
        results = []
        valid_graphs = []
        valid_indices = []
        
        # 解析和构建图
        for i, expression_str in enumerate(expressions):
            try:
                parsed_result = self.parse_expression(expression_str)
                ast = self.expression_parser.parse_expression(
                    parsed_result['expression'], 
                    parsed_result['variable_mapping']
                )
                graph_data = self.graph_builder.ast_to_graph(ast)
                valid_graphs.append(graph_data)
                valid_indices.append(i)
                
                # 创建结果占位符
                results.append({
                    'expression': expression_str,
                    'parsed_expression': parsed_result['expression'],
                    'variable_mapping': parsed_result['variable_mapping']
                })
                
            except Exception as e:
                results.append({
                    'expression': expression_str,
                    'error': str(e),
                    'predicted_sharpe': None
                })
        
        # 批量预测有效的图
        if valid_graphs:
            try:
                from torch_geometric.data import Batch
                batch = Batch.from_data_list(valid_graphs)
                batch = batch.to(self.device)
                
                with torch.no_grad():
                    predictions = self.model(batch)
                    sharpe_values = predictions.cpu().numpy()
                
                # 将预测结果填入对应位置
                for idx, valid_idx in enumerate(valid_indices):
                    results[valid_idx]['predicted_sharpe'] = float(sharpe_values[idx, 0])
                    
            except Exception as e:
                # 如果批量预测失败，标记所有有效结果为错误
                for valid_idx in valid_indices:
                    results[valid_idx]['error'] = f"批量预测失败: {e}"
                    results[valid_idx]['predicted_sharpe'] = None
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        return {
            'model_path': self.model_path,
            'device': str(self.device),
            'feature_dims': self.feature_dims,
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'model_config': {
                'hidden_dim': self.model.hidden_dim,
                'num_layers': self.model.num_layers,
                'dropout': self.model.dropout,
                'pool_method': self.model.pool_method,
                'use_edge_features': self.model.use_edge_features,
                'use_graph_features': self.model.use_graph_features,
                'use_attention': self.model.use_attention
            }
        }
    
    def list_available_models(self) -> List[str]:
        """
        列出所有可用模型
        
        Returns:
            模型描述列表
        """
        return self.model_manager.list_available_models()


def main():
    """测试AlphaPredictor功能"""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("=== AlphaPredictor测试 ===")
    
    try:
        # 初始化预测器
        predictor = AlphaPredictor()
        
        # 显示模型信息
        model_info = predictor.get_model_info()
        print(f"\n模型信息:")
        print(f"  路径: {model_info['model_path']}")
        print(f"  设备: {model_info['device']}")
        print(f"  参数数量: {model_info['model_parameters']:,}")
        
        # 测试单个表达式预测
        test_expression = "sub(ts_norm(x2,1110),ts_norm(x1,1110));2 TCFBaseClean 5 TCFBaseClean 2"
        print(f"\n测试表达式: {test_expression}")
        
        result = predictor.predict_single(test_expression)
        if 'error' in result:
            print(f"预测错误: {result['error']}")
        else:
            print(f"预测Sharpe值: {result['predicted_sharpe']:.6f}")
            print(f"变量映射: {result['variable_mapping']}")
        
    except Exception as e:
        print(f"测试失败: {e}")


if __name__ == "__main__":
    main()