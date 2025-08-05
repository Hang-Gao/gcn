#!/usr/bin/env python3
"""
Alpha表达式GCN评估器主程序
Author: hang
Description: 使用GCN预测alpha表达式的Sharpe值的完整程序

核心特点：
1. 理解x1,x2,x3变量的动态映射机制 - 每个表达式都有独特的变量到特征的映射
2. 使用IS数据集训练，OS数据集测试
3. 完整的端到端流程：数据加载→表达式解析→图转换→模型训练→性能评估
"""

import os
import sys
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Any, Tuple
import json
import numpy as np

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 导入自定义模块
from data.csv_parser import CSVDataParser
from data.dataset import AlphaDataset
from parser.expression_parser import AlphaExpressionParser
from model import GraphBuilder, create_alpha_gcn_model
from model.trainer import AlphaGCNTrainer
# from utils.visualization import Visualizer
# from utils.metrics import MetricsCalculator


def setup_logging(log_level: str = "INFO", log_file: str = None) -> None:
    """设置日志配置"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 配置根logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # 如果指定了日志文件，添加文件handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)
    
    logger = logging.getLogger(__name__)
    logger.info(f"日志系统初始化完成，级别: {log_level}")
    return logger


class AlphaGCNEvaluator:
    """Alpha表达式GCN评估器主类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化评估器
        
        Args:
            config: 配置字典，包含所有超参数和路径
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 设置设备
        self.device = torch.device(config.get('device', 'cpu'))
        self.logger.info(f"使用设备: {self.device}")
        
        # 初始化组件
        self.csv_parser = CSVDataParser()
        self.expression_parser = AlphaExpressionParser()
        self.graph_builder = None  # 稍后根据词汇表初始化
        self.model = None
        self.trainer = None
        # self.visualizer = Visualizer()
        # self.metrics_calculator = MetricsCalculator()
        
        # 数据容器
        self.train_data = None
        self.test_data = None
        self.train_dataset = None
        self.test_dataset = None
        
        # 结果存储
        self.training_results = {}
        self.evaluation_results = {}
    
    def load_data(self) -> None:
        """加载和预处理数据"""
        self.logger.info("开始加载数据...")

        # 加载训练数据 (IS)
        train_file = self.config['data']['train_file']
        if not os.path.exists(train_file):
            self.logger.error(f"训练文件不存在: {train_file}")
            raise FileNotFoundError(f"训练文件不存在: {train_file}")
        self.logger.info(f"加载训练数据: {train_file}")
        self.train_data = self.csv_parser.parse_dataset(train_file)
        self.logger.info(f"训练数据加载完成: {len(self.train_data)} 个样本")

        # 加载测试数据 (OS)
        test_file = self.config['data']['test_file']
        if not os.path.exists(test_file):
            self.logger.error(f"测试文件不存在: {test_file}")
            raise FileNotFoundError(f"测试文件不存在: {test_file}")
        self.logger.info(f"加载测试数据: {test_file}")
        self.test_data = self.csv_parser.parse_dataset(test_file)
        self.logger.info(f"测试数据加载完成: {len(self.test_data)} 个样本")

        # 数据验证
        self._validate_data()

        # 数据清洗
        self.train_data = self.csv_parser.clean_data(self.train_data)
        self.test_data = self.csv_parser.clean_data(self.test_data)
        
        self.logger.info(f"数据清洗后 - 训练: {len(self.train_data)}, 测试: {len(self.test_data)}")
    
    def _validate_data(self) -> None:
        """验证数据质量"""
        self.logger.info("验证数据质量...")
        
        # 验证训练数据
        train_validation = self.csv_parser.validate_data(self.train_data)
        self.logger.info(f"训练数据验证率: {train_validation['validation_rate']:.2%}")
        
        # 验证测试数据
        test_validation = self.csv_parser.validate_data(self.test_data)
        self.logger.info(f"测试数据验证率: {test_validation['validation_rate']:.2%}")
        
        # 保存验证报告
        validation_report = {
            'train_validation': train_validation,
            'test_validation': test_validation,
            'timestamp': datetime.now().isoformat()
        }
        
        report_file = os.path.join(self.config['output']['results_dir'], 'data_validation_report.json')
        os.makedirs(os.path.dirname(report_file), exist_ok=True)
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(validation_report, f, indent=2, ensure_ascii=False)
    
    def parse_expressions(self) -> None:
        """解析所有表达式为AST"""
        self.logger.info("开始解析表达式...")
        
        # 解析训练集表达式
        self.logger.info("解析训练集表达式...")
        train_asts = self.expression_parser.parse_multiple_expressions(self.train_data)
        
        # 解析测试集表达式
        self.logger.info("解析测试集表达式...")
        test_asts = self.expression_parser.parse_multiple_expressions(self.test_data)
        
        # 获取解析统计信息
        parse_stats = self.expression_parser.get_parse_statistics()
        self.logger.info(f"表达式解析统计: {parse_stats}")
        
        # 构建词汇表（基于训练集）
        self.logger.info("构建图转换器...")
        self.graph_builder = GraphBuilder()
        self.graph_builder.build_vocabulary(train_asts)
        
        vocab_size = self.graph_builder.get_vocabulary_size()
        self.logger.info(f"词汇表大小: {vocab_size}")
        
        # 将AST和数据组合
        self._combine_asts_with_data(train_asts, test_asts)
    
    def _combine_asts_with_data(self, train_asts: List, test_asts: List) -> None:
        """将AST与原始数据组合"""
        # 为训练数据添加AST
        for i, ast in enumerate(train_asts):
            if i < len(self.train_data):
                self.train_data[i]['ast'] = ast
        
        # 为测试数据添加AST
        for i, ast in enumerate(test_asts):
            if i < len(self.test_data):
                self.test_data[i]['ast'] = ast
        
        # 过滤掉没有有效AST的数据
        self.train_data = [d for d in self.train_data if 'ast' in d and d['ast'] is not None]
        self.test_data = [d for d in self.test_data if 'ast' in d and d['ast'] is not None]
        
        self.logger.info(f"AST组合后 - 训练: {len(self.train_data)}, 测试: {len(self.test_data)}")
    
    def create_datasets(self) -> None:
        """创建PyTorch数据集"""
        self.logger.info("创建数据集...")
        
        # 创建训练数据集
        self.train_dataset = AlphaDataset(
            self.train_data, 
            self.graph_builder,
            transform_graphs=True
        )
        
        # 创建测试数据集
        self.test_dataset = AlphaDataset(
            self.test_data,
            self.graph_builder,
            transform_graphs=True
        )
        
        self.logger.info(f"数据集创建完成 - 训练: {len(self.train_dataset)}, 测试: {len(self.test_dataset)}")
        
        # 获取图统计信息
        train_stats = self.train_dataset.get_graph_statistics()
        self.logger.info(f"训练集图统计: {train_stats}")
    
    def initialize_model(self) -> None:
        """初始化模型"""
        self.logger.info("初始化模型...")
        
        # 获取特征维度
        # 从样本图中动态获取维度，比从GraphBuilder中获取更准确
        sample_graph = self.train_dataset[0] 
        feature_dims = {
            'node_feature_dim': sample_graph.x.shape[1],
            'edge_feature_dim': sample_graph.edge_attr.shape[1] if hasattr(sample_graph, 'edge_attr') and sample_graph.edge_attr is not None else 2,
            'graph_feature_dim': sample_graph.graph_features.shape[0] if hasattr(sample_graph, 'graph_features') and hasattr(sample_graph.graph_features, 'shape') else 7
        }
        
        # 创建模型
        model_config = self.config['model']
        
        # 使用在 gcn_model.py 中定义的工厂函数来创建模型
        # 这能确保模型创建逻辑的统一，并方便未来扩展（例如，支持集成模型或不同类型的GCN）
        self.model = create_alpha_gcn_model(
            feature_dims=feature_dims,
            config=model_config
        ).to(self.device)
        
        # 模型信息
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"模型初始化完成:")
        self.logger.info(f"  节点特征维度: {feature_dims['node_feature_dim']}")
        self.logger.info(f"  隐藏维度: {model_config.get('hidden_dim', 128)}")
        self.logger.info(f"  网络层数: {model_config.get('num_layers', 3)}")
        self.logger.info(f"  总参数量: {total_params:,}")
        self.logger.info(f"  可训练参数: {trainable_params:,}")
    
    def train_model(self) -> None:
        """训练模型"""
        self.logger.info("开始训练模型...")
        
        # 创建训练器
        train_config = self.config['training']
        self.trainer = AlphaGCNTrainer(
            model=self.model,
            device=self.device,
            learning_rate=train_config.get('learning_rate', 0.001),
            weight_decay=train_config.get('weight_decay', 1e-5),
            scheduler_type=train_config.get('scheduler', 'plateau')
        )
        
        # 从训练集中分离出验证集
        train_size = len(self.train_dataset)
        val_ratio = train_config.get('val_ratio', 0.1)  # 默认使用10%作为验证集
        val_size = int(train_size * val_ratio)
        train_size = train_size - val_size
        
        # 设置随机种子确保可重现性
        np.random.seed(42)
        indices = np.random.permutation(len(self.train_dataset))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # 创建训练和验证子集
        train_subset = torch.utils.data.Subset(self.train_dataset, train_indices)
        val_subset = torch.utils.data.Subset(self.train_dataset, val_indices)
        
        self.logger.info(f"数据分割完成:")
        self.logger.info(f"  原始训练集: {len(self.train_dataset)} 样本")
        self.logger.info(f"  实际训练集: {len(train_subset)} 样本")
        self.logger.info(f"  验证集: {len(val_subset)} 样本")
        self.logger.info(f"  测试集: {len(self.test_dataset)} 样本 (完全独立)")
        
        # 训练模型（注意：trainer.train方法需要的是Dataset而不是DataLoader）
        self.training_results = self.trainer.train(
            train_dataset=train_subset,
            val_dataset=val_subset,
            epochs=train_config.get('epochs', 100),
            batch_size=train_config.get('batch_size', 32)
        )
        
        self.logger.info("模型训练完成")
        
        # 保存模型
        self._save_model()
    
    def _save_model(self) -> None:
        """保存训练好的模型"""
        model_dir = os.path.join(self.config['output']['results_dir'], 'model_checkpoints')
        os.makedirs(model_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(model_dir, f'alpha_gcn_model_{timestamp}.pth')
        
        # 保存模型状态
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'training_results': self.training_results,
            'vocab_size': self.graph_builder.get_vocabulary_size(),
            'timestamp': timestamp
        }, model_path)
        
        self.logger.info(f"模型已保存至: {model_path}")
    
    def evaluate_model(self) -> None:
        """评估模型在测试集上的性能"""
        self.logger.info("开始评估模型...")
        
        # 使用训练器的评估方法
        self.evaluation_results = self.trainer.evaluate(
            self.test_dataset,
            batch_size=self.config['training'].get('batch_size', 32)
        )
        
        metrics = self.evaluation_results['metrics']
        self.logger.info(f"模型评估结果:")
        self.logger.info(f"  样本数量: {self.evaluation_results['num_samples']}")
        self.logger.info(f"  MSE: {metrics['mse']:.6f}")
        self.logger.info(f"  MAE: {metrics['mae']:.6f}")
        self.logger.info(f"  RMSE: {metrics['rmse']:.6f}")
        self.logger.info(f"  R²: {metrics['r2']:.6f}")
        self.logger.info(f"  相关性: {metrics['correlation']:.6f}")
    
    def generate_visualizations(self) -> None:
        """生成结果可视化"""
        self.logger.info("生成可视化图表...")
        
        viz_dir = os.path.join(self.config['output']['results_dir'], 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 训练损失曲线（使用trainer的内置方法）
        if self.trainer and hasattr(self.trainer, 'plot_training_history'):
            self.trainer.plot_training_history(
                save_path=os.path.join(viz_dir, f'training_curves_{timestamp}.png')
            )
        
        # 2. 预测vs真实值散点图（使用trainer的内置方法）
        if self.evaluation_results:
            from model.trainer import plot_predictions
            plot_predictions(
                self.evaluation_results['predictions'],
                self.evaluation_results['targets'],
                title="Alpha GCN 预测 vs 真实值",
                save_path=os.path.join(viz_dir, f'predictions_vs_true_{timestamp}.png')
            )
        
        self.logger.info(f"可视化图表已保存至: {viz_dir}")
    
    def save_results(self) -> None:
        """保存所有结果"""
        self.logger.info("保存结果...")
        
        results_dir = self.config['output']['results_dir']
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存完整结果
        full_results = {
            'config': self.config,
            'training_results': self.training_results,
            'evaluation_results': self.evaluation_results,
            'data_info': {
                'train_samples': len(self.train_data) if self.train_data else 0,
                'test_samples': len(self.test_data) if self.test_data else 0,
                'vocab_size': self.graph_builder.get_vocabulary_size() if self.graph_builder else 0
            },
            'timestamp': timestamp
        }
        
        # 转换numpy类型为Python原生类型，以便JSON序列化
        def convert_numpy_types(obj):
            if isinstance(obj, np.float32):
                return float(obj)
            elif isinstance(obj, np.float64):
                return float(obj)
            elif isinstance(obj, np.int32):
                return int(obj)
            elif isinstance(obj, np.int64):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        full_results_clean = convert_numpy_types(full_results)
        
        results_file = os.path.join(results_dir, f'full_results_{timestamp}.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(full_results_clean, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"结果已保存至: {results_file}")
    
    def run_complete_pipeline(self) -> None:
        """运行完整的训练和评估流程"""
        self.logger.info("开始运行完整流程...")
        
        try:
            # 1. 数据加载
            self.load_data()
            
            # 2. 表达式解析
            self.parse_expressions()
            
            # 3. 创建数据集
            self.create_datasets()
            
            # 4. 初始化模型
            self.initialize_model()
            
            # 5. 训练模型
            self.train_model()
            
            # 6. 评估模型
            self.evaluate_model()
            
            # 7. 生成可视化
            self.generate_visualizations()
            
            # 8. 保存结果
            self.save_results()
            
            self.logger.info("完整流程执行成功！")
            
        except Exception as e:
            self.logger.error(f"流程执行失败: {str(e)}", exc_info=True)
            raise


def create_default_config() -> Dict[str, Any]:
    """创建默认配置"""
    return {
        'data': {
            'train_file': 'alpha_OS_train_400.csv',
            'test_file': 'alpha_OS_test_139.csv'
        },
        'model': {
            'hidden_dim': 128,
            'num_layers': 3,
            'dropout': 0.2
        },
        'training': {
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'scheduler': 'ReduceLROnPlateau',
            'val_ratio': 0.1  # 验证集比例：10%
        },
        'output': {
            'results_dir': 'results',
            'log_level': 'INFO'
        },
        'device': 'cpu'  # 默认使用CPU，确保兼容性
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Alpha表达式GCN评估器')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--train-file', type=str, default='alpha_OS_train_400.csv', help='训练数据文件')
    parser.add_argument('--test-file', type=str, default='alpha_OS_test_139.csv', help='测试数据文件')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--hidden-dim', type=int, default=128, help='隐藏层维度')
    parser.add_argument('--device', type=str, default='cpu', help='计算设备')
    parser.add_argument('--log-level', type=str, default='INFO', help='日志级别')
    
    args = parser.parse_args()
    
    # 创建配置
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        config = create_default_config()
    
    # 命令行参数覆盖配置
    config['data']['train_file'] = args.train_file
    config['data']['test_file'] = args.test_file
    config['training']['epochs'] = args.epochs
    config['training']['batch_size'] = args.batch_size
    config['model']['hidden_dim'] = args.hidden_dim
    config['device'] = args.device
    config['output']['log_level'] = args.log_level
    
    # 设置日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(config['output']['results_dir'], 'logs', f'main_{timestamp}.log')
    logger = setup_logging(config['output']['log_level'], log_file)
    
    logger.info("=" * 80)
    logger.info("Alpha表达式GCN评估器启动")
    logger.info("=" * 80)
    logger.info(f"配置信息: {json.dumps(config, indent=2, ensure_ascii=False)}")
    
    try:
        # 创建评估器并运行
        evaluator = AlphaGCNEvaluator(config)
        evaluator.run_complete_pipeline()
        
        # 输出最终结果摘要
        if evaluator.evaluation_results:
            metrics = evaluator.evaluation_results['metrics']
            logger.info("=" * 80)
            logger.info("最终评估结果摘要:")
            logger.info(f"  样本数量: {evaluator.evaluation_results['num_samples']}")
            logger.info(f"  MSE: {metrics['mse']:.6f}")
            logger.info(f"  MAE: {metrics['mae']:.6f}")
            logger.info(f"  RMSE: {metrics['rmse']:.6f}")
            logger.info(f"  R²: {metrics['r2']:.6f}")
            logger.info("=" * 80)
        
        logger.info("程序执行完成！")
        
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()