import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import logging
import time
import os
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns

# 配置matplotlib支持中文显示和负号问题
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Hiragino Sans GB', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['mathtext.default'] = 'regular'  # 使用常规字体渲染数学文本
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

from .gcn_model import AlphaGCN, AlphaGCNEnsemble

logger = logging.getLogger(__name__)


class AlphaGCNTrainer:
    """
    Alpha GCN训练器
    
    负责模型训练、验证和评估
    """
    
    def __init__(self, 
                 model: nn.Module,
                 device: torch.device = None,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-5,
                 scheduler_type: str = 'step',
                 scheduler_params: Dict[str, Any] = None):
        """
        初始化训练器
        
        Args:
            model: GCN模型
            device: 计算设备
            learning_rate: 学习率
            weight_decay: 权重衰减
            scheduler_type: 学习率调度器类型 ('step', 'plateau')
            scheduler_params: 调度器参数
        """
        self.model = model
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
        # 学习率调度器
        if scheduler_params is None:
            scheduler_params = {}
            
        if scheduler_type == 'step':
            self.scheduler = StepLR(
                self.optimizer,
                step_size=scheduler_params.get('step_size', 20),
                gamma=scheduler_params.get('gamma', 0.5)
            )
        elif scheduler_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_params.get('factor', 0.5),
                patience=scheduler_params.get('patience', 10),
                min_lr=scheduler_params.get('min_lr', 1e-6)
            )
        else:
            self.scheduler = None
        
        # 训练历史
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'lr': []
        }
        
        logger.info(f"训练器初始化完成")
        logger.info(f"  设备: {self.device}")
        logger.info(f"  学习率: {learning_rate}")
        logger.info(f"  权重衰减: {weight_decay}")
        logger.info(f"  调度器: {scheduler_type}")
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            
        Returns:
            (平均损失, 训练指标)
        """
        self.model.train()
        total_loss = 0.0
        predictions = []
        targets = []
        
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(batch)
            
            # 计算损失
            loss = self.criterion(output.squeeze(), batch.y)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            predictions.extend(output.detach().cpu().numpy().flatten())
            targets.extend(batch.y.detach().cpu().numpy().flatten())
            
            if (batch_idx + 1) % 20 == 0:
                logger.debug(f"批次 {batch_idx + 1}/{len(train_loader)}, 损失: {loss.item():.6f}")
        
        avg_loss = total_loss / len(train_loader)
        metrics = self._calculate_metrics(predictions, targets)
        
        return avg_loss, metrics
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        验证一个epoch
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            (平均损失, 验证指标)
        """
        self.model.eval()
        total_loss = 0.0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                
                # 前向传播
                output = self.model(batch)
                loss = self.criterion(output.squeeze(), batch.y)
                
                # 统计
                total_loss += loss.item()
                predictions.extend(output.cpu().numpy().flatten())
                targets.extend(batch.y.cpu().numpy().flatten())
        
        avg_loss = total_loss / len(val_loader)
        metrics = self._calculate_metrics(predictions, targets)
        
        return avg_loss, metrics
    
    def train(self, 
              train_dataset,
              val_dataset = None,
              epochs: int = 100,
              batch_size: int = 32,
              save_path: str = None,
              patience: int = 20,
              verbose: bool = True) -> Dict[str, Any]:
        """
        完整训练流程
        
        Args:
            train_dataset: 训练数据集
            val_dataset: 验证数据集
            epochs: 训练轮数
            batch_size: 批次大小
            save_path: 模型保存路径
            patience: 早停耐心值
            verbose: 是否显示详细信息
            
        Returns:
            训练结果字典
        """
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None
        
        # 早停相关
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        
        # 开始训练
        start_time = time.time()
        logger.info(f"开始训练，共 {epochs} 轮")
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # 训练
            train_loss, train_metrics = self.train_epoch(train_loader)
            
            # 验证
            if val_loader:
                val_loss, val_metrics = self.validate_epoch(val_loader)
            else:
                val_loss, val_metrics = train_loss, train_metrics
            
            # 更新学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # 记录历史
            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['train_metrics'].append(train_metrics)
            self.train_history['val_metrics'].append(val_metrics)
            self.train_history['lr'].append(current_lr)
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                
                # 保存最佳模型
                if save_path:
                    self.save_model(save_path)
            else:
                patience_counter += 1
            
            # 输出进度
            if verbose and (epoch + 1) % 5 == 0:
                epoch_time = time.time() - epoch_start
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} ({epoch_time:.2f}s) - "
                    f"训练损失: {train_loss:.6f}, 验证损失: {val_loss:.6f}, "
                    f"R^2: {val_metrics['r2']:.4f}, 学习率: {current_lr:.2e}"
                )
            
            # 早停
            if patience_counter >= patience:
                logger.info(f"早停触发，在第 {epoch + 1} 轮停止训练")
                break
        
        total_time = time.time() - start_time
        
        # 训练结果
        result = {
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'total_epochs': epoch + 1,
            'total_time': total_time,
            'final_train_loss': train_loss,
            'final_val_loss': val_loss,
            'final_metrics': val_metrics,
            'history': self.train_history
        }
        
        logger.info(f"训练完成！")
        logger.info(f"  最佳轮次: {best_epoch + 1}")
        logger.info(f"  最佳验证损失: {best_val_loss:.6f}")
        logger.info(f"  最终R^2: {val_metrics['r2']:.4f}")
        logger.info(f"  总训练时间: {total_time:.2f}秒")
        
        return result
    
    def evaluate(self, test_dataset, batch_size: int = 32) -> Dict[str, Any]:
        """
        评估模型
        
        Args:
            test_dataset: 测试数据集
            batch_size: 批次大小
            
        Returns:
            评估结果字典
        """
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        self.model.eval()
        predictions = []
        targets = []
        graph_info = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                
                # 前向传播
                output = self.model(batch)
                
                # 收集结果
                batch_preds = output.cpu().numpy().flatten()
                batch_targets = batch.y.cpu().numpy().flatten()
                
                predictions.extend(batch_preds)
                targets.extend(batch_targets)
                
                # 收集图信息（用于分析）
                for i in range(len(batch_preds)):
                    if hasattr(batch, 'expression_string'):
                        graph_info.append({
                            'prediction': batch_preds[i],
                            'target': batch_targets[i],
                            'expression': batch.expression_string[i] if isinstance(batch.expression_string, list) else 'N/A'
                        })
        
        # 计算指标
        metrics = self._calculate_metrics(predictions, targets)
        
        # 误差分析
        errors = np.array(predictions) - np.array(targets)
        error_stats = {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'max_error': np.max(np.abs(errors)),
            'error_percentiles': {
                '25%': np.percentile(np.abs(errors), 25),
                '50%': np.percentile(np.abs(errors), 50),
                '75%': np.percentile(np.abs(errors), 75),
                '90%': np.percentile(np.abs(errors), 90),
                '95%': np.percentile(np.abs(errors), 95)
            }
        }
        
        result = {
            'metrics': metrics,
            'error_stats': error_stats,
            'predictions': predictions,
            'targets': targets,
            'graph_info': graph_info,
            'num_samples': len(predictions)
        }
        
        logger.info(f"模型评估完成:")
        logger.info(f"  样本数: {len(predictions)}")
        logger.info(f"  MSE: {metrics['mse']:.6f}")
        logger.info(f"  MAE: {metrics['mae']:.6f}")
        logger.info(f"  R^2: {metrics['r2']:.4f}")
        logger.info(f"  RMSE: {metrics['rmse']:.6f}")
        
        return result
    
    def _calculate_metrics(self, predictions: List[float], targets: List[float]) -> Dict[str, float]:
        """
        计算评估指标
        
        Args:
            predictions: 预测值列表
            targets: 真实值列表
            
        Returns:
            指标字典
        """
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        rmse = np.sqrt(mse)
        
        # 相关系数
        correlation = np.corrcoef(predictions, targets)[0, 1]
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'rmse': rmse,
            'correlation': correlation
        }
    
    def save_model(self, path: str):
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_history': self.train_history,
            'model_config': getattr(self.model, 'config', None)
        }
        
        torch.save(checkpoint, path)
        logger.info(f"模型已保存到: {path}")
    
    def load_model(self, path: str):
        """
        加载模型
        
        Args:
            path: 模型路径
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'train_history' in checkpoint:
            self.train_history = checkpoint['train_history']
        
        logger.info(f"模型已从 {path} 加载")
    
    def plot_training_history(self, save_path: str = None):
        """
        绘制训练历史
        
        Args:
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        axes[0, 0].plot(self.train_history['train_loss'], label='训练损失')
        axes[0, 0].plot(self.train_history['val_loss'], label='验证损失')
        axes[0, 0].set_title('损失曲线')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # R^2曲线
        train_r2 = [m['r2'] for m in self.train_history['train_metrics']]
        val_r2 = [m['r2'] for m in self.train_history['val_metrics']]
        axes[0, 1].plot(train_r2, label='训练R^2')
        axes[0, 1].plot(val_r2, label='验证R^2')
        axes[0, 1].set_title('R^2曲线')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('R^2')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 学习率曲线
        axes[1, 0].plot(self.train_history['lr'])
        axes[1, 0].set_title('学习率曲线')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
        
        # RMSE曲线
        train_rmse = [m['rmse'] for m in self.train_history['train_metrics']]
        val_rmse = [m['rmse'] for m in self.train_history['val_metrics']]
        axes[1, 1].plot(train_rmse, label='训练RMSE')
        axes[1, 1].plot(val_rmse, label='验证RMSE')
        axes[1, 1].set_title('RMSE曲线')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('RMSE')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"训练历史图已保存到: {save_path}")
        
        plt.close()  # 关闭图形以释放内存


def plot_predictions(predictions: List[float], 
                    targets: List[float],
                    title: str = "预测 vs 真实值",
                    save_path: str = None):
    """
    绘制预测vs真实值散点图
    
    Args:
        predictions: 预测值列表
        targets: 真实值列表
        title: 图标题
        save_path: 保存路径
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # 散点图
    ax.scatter(targets, predictions, alpha=0.6, s=30)
    
    # 对角线
    min_val = min(min(targets), min(predictions))
    max_val = max(max(targets), max(predictions))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='完美预测')
    
    # 计算指标
    mse = mean_squared_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    correlation = np.corrcoef(predictions, targets)[0, 1]
    
    ax.set_xlabel('真实值')
    ax.set_ylabel('预测值')
    ax.set_title(f'{title}\nMSE: {mse:.6f}, R^2: {r2:.4f}, 相关性: {correlation:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"预测图已保存到: {save_path}")
    
    plt.close()  # 关闭图形以释放内存