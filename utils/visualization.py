"""
可视化工具模块
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any
from sklearn.metrics import mean_squared_error, r2_score
import logging

logger = logging.getLogger(__name__)

# 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Hiragino Sans GB', 'Arial Unicode MS', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def plot_predictions(predictions: List[float], 
                    targets: List[float],
                    title: str = "预测 vs 真实值",
                    save_path: Optional[str] = None,
                    show_plot: bool = True,
                    figsize: tuple = (10, 8)) -> None:
    """
    绘制预测vs真实值散点图
    
    Args:
        predictions: 预测值列表
        targets: 真实值列表
        title: 图标题
        save_path: 保存路径
        show_plot: 是否显示图形
        figsize: 图形大小
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # 散点图
    ax.scatter(targets, predictions, alpha=0.6, s=30, color='blue', edgecolors='black', linewidth=0.5)
    
    # 对角线（完美预测线）
    min_val = min(np.min(targets), np.min(predictions))
    max_val = max(np.max(targets), np.max(predictions))
    margin = (max_val - min_val) * 0.05
    
    ax.plot([min_val - margin, max_val + margin], 
            [min_val - margin, max_val + margin], 
            'r--', linewidth=2, label='完美预测', alpha=0.8)
    
    # 计算指标
    mse = mean_squared_error(targets, predictions)
    mae = np.mean(np.abs(predictions - targets))
    r2 = r2_score(targets, predictions)
    correlation = np.corrcoef(predictions, targets)[0, 1]
    rmse = np.sqrt(mse)
    
    # 添加统计信息
    stats_text = f'样本数: {len(predictions)}\n'
    stats_text += f'MSE: {mse:.6f}\n'
    stats_text += f'MAE: {mae:.6f}\n'
    stats_text += f'RMSE: {rmse:.6f}\n'
    stats_text += f'R²: {r2:.4f}\n'
    stats_text += f'相关性: {correlation:.4f}'
    
    # 在图上添加文本框
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel('真实Sharpe值')
    ax.set_ylabel('预测Sharpe值')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 设置坐标轴范围
    ax.set_xlim(min_val - margin, max_val + margin)
    ax.set_ylim(min_val - margin, max_val + margin)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"预测散点图已保存到: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_training_curves(train_history: Dict[str, List],
                        save_path: Optional[str] = None,
                        show_plot: bool = True,
                        figsize: tuple = (15, 10)) -> None:
    """
    绘制训练过程曲线
    
    Args:
        train_history: 训练历史字典
        save_path: 保存路径
        show_plot: 是否显示图形
        figsize: 图形大小
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    epochs = range(1, len(train_history['train_loss']) + 1)
    
    # 损失曲线
    axes[0, 0].plot(epochs, train_history['train_loss'], 'b-', label='训练损失', linewidth=2)
    axes[0, 0].plot(epochs, train_history['val_loss'], 'r-', label='验证损失', linewidth=2)
    axes[0, 0].set_title('损失曲线')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss (MSE)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # R²曲线
    train_r2 = [m['r2'] for m in train_history['train_metrics']]
    val_r2 = [m['r2'] for m in train_history['val_metrics']]
    axes[0, 1].plot(epochs, train_r2, 'b-', label='训练R²', linewidth=2)
    axes[0, 1].plot(epochs, val_r2, 'r-', label='验证R²', linewidth=2)
    axes[0, 1].set_title('R²曲线')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('R²')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 学习率曲线
    if 'lr' in train_history and train_history['lr']:
        axes[1, 0].plot(epochs, train_history['lr'], 'g-', linewidth=2)
        axes[1, 0].set_title('学习率曲线')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
    
    # MAE曲线
    train_mae = [m['mae'] for m in train_history['train_metrics']]
    val_mae = [m['mae'] for m in train_history['val_metrics']]
    axes[1, 1].plot(epochs, train_mae, 'b-', label='训练MAE', linewidth=2)
    axes[1, 1].plot(epochs, val_mae, 'r-', label='验证MAE', linewidth=2)
    axes[1, 1].set_title('MAE曲线')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('MAE')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"训练曲线图已保存到: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_error_distribution(predictions: List[float],
                           targets: List[float],
                           title: str = "预测误差分布",
                           save_path: Optional[str] = None,
                           show_plot: bool = True,
                           figsize: tuple = (15, 5)) -> None:
    """
    绘制预测误差分布图
    
    Args:
        predictions: 预测值列表
        targets: 真实值列表
        title: 图标题
        save_path: 保存路径
        show_plot: 是否显示图形
        figsize: 图形大小
    """
    predictions = np.array(predictions)
    targets = np.array(targets)
    errors = predictions - targets
    abs_errors = np.abs(errors)
    relative_errors = abs_errors / (np.abs(targets) + 1e-8)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # 误差直方图
    axes[0].hist(errors, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0].axvline(np.mean(errors), color='red', linestyle='--', linewidth=2, label=f'均值: {np.mean(errors):.6f}')
    axes[0].axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5, label='零误差')
    axes[0].set_title('预测误差分布')
    axes[0].set_xlabel('预测误差')
    axes[0].set_ylabel('频数')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 绝对误差直方图
    axes[1].hist(abs_errors, bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[1].axvline(np.mean(abs_errors), color='red', linestyle='--', linewidth=2, 
                   label=f'MAE: {np.mean(abs_errors):.6f}')
    axes[1].set_title('绝对误差分布')
    axes[1].set_xlabel('绝对误差')
    axes[1].set_ylabel('频数')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 相对误差直方图
    axes[2].hist(relative_errors, bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[2].axvline(np.mean(relative_errors), color='red', linestyle='--', linewidth=2,
                   label=f'平均相对误差: {np.mean(relative_errors):.4f}')
    axes[2].set_title('相对误差分布')
    axes[2].set_xlabel('相对误差')
    axes[2].set_ylabel('频数')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # 添加统计信息
    stats_text = f"误差统计:\n"
    stats_text += f"  均值误差: {np.mean(errors):.6f}\n"
    stats_text += f"  标准差: {np.std(errors):.6f}\n"
    stats_text += f"  最大绝对误差: {np.max(abs_errors):.6f}\n"
    stats_text += f"  90%分位数: {np.percentile(abs_errors, 90):.6f}"
    
    print(stats_text)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"误差分布图已保存到: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_performance_analysis(evaluation_result: Dict[str, Any],
                             title: str = "模型性能综合分析",
                             save_path: Optional[str] = None,
                             show_plot: bool = True,
                             figsize: tuple = (20, 15)) -> None:
    """
    绘制综合性能分析图
    
    Args:
        evaluation_result: 评估结果字典
        title: 图标题
        save_path: 保存路径
        show_plot: 是否显示图形
        figsize: 图形大小
    """
    predictions = np.array(evaluation_result['predictions'])
    targets = np.array(evaluation_result['targets'])
    metrics = evaluation_result['metrics']
    error_stats = evaluation_result['error_stats']
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. 预测vs真实值散点图
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(targets, predictions, alpha=0.6, s=20)
    min_val = min(np.min(targets), np.min(predictions))
    max_val = max(np.max(targets), np.max(predictions))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax1.set_xlabel('真实值')
    ax1.set_ylabel('预测值')
    ax1.set_title(f"预测vs真实值\nR²={metrics['r2']:.4f}")
    ax1.grid(True, alpha=0.3)
    
    # 2. 残差图
    ax2 = fig.add_subplot(gs[0, 1])
    residuals = predictions - targets
    ax2.scatter(targets, residuals, alpha=0.6, s=20)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('真实值')
    ax2.set_ylabel('残差')
    ax2.set_title('残差图')
    ax2.grid(True, alpha=0.3)
    
    # 3. Q-Q图（正态性检验）
    ax3 = fig.add_subplot(gs[0, 2])
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.set_title('残差Q-Q图')
    ax3.grid(True, alpha=0.3)
    
    # 4. 误差分布直方图
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(residuals, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax4.axvline(0, color='red', linestyle='--', linewidth=2)
    ax4.set_xlabel('残差')
    ax4.set_ylabel('频数')
    ax4.set_title('残差分布')
    ax4.grid(True, alpha=0.3)
    
    # 5. 绝对误差分布
    ax5 = fig.add_subplot(gs[1, 1])
    abs_errors = np.abs(residuals)
    ax5.hist(abs_errors, bins=30, alpha=0.7, color='orange', edgecolor='black')
    ax5.set_xlabel('绝对误差')
    ax5.set_ylabel('频数')
    ax5.set_title(f'绝对误差分布\nMAE={metrics["mae"]:.6f}')
    ax5.grid(True, alpha=0.3)
    
    # 6. 误差vs预测值
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.scatter(predictions, abs_errors, alpha=0.6, s=20)
    ax6.set_xlabel('预测值')
    ax6.set_ylabel('绝对误差')
    ax6.set_title('误差vs预测值')
    ax6.grid(True, alpha=0.3)
    
    # 7. 指标总结表
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('tight')
    ax7.axis('off')
    
    # 创建指标表格
    metrics_data = [
        ['指标', '数值', '说明'],
        ['MSE', f"{metrics['mse']:.6f}", '均方误差'],
        ['MAE', f"{metrics['mae']:.6f}", '平均绝对误差'],
        ['RMSE', f"{metrics['rmse']:.6f}", '均方根误差'],
        ['R²', f"{metrics['r2']:.4f}", '决定系数'],
        ['相关性', f"{metrics['correlation']:.4f}", 'Pearson相关系数'],
        ['样本数', f"{len(predictions)}", '测试样本总数'],
        ['最大误差', f"{error_stats['max_error']:.6f}", '最大绝对误差'],
        ['误差标准差', f"{error_stats['std_error']:.6f}", '误差分布的标准差']
    ]
    
    table = ax7.table(cellText=metrics_data, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    # 设置表格样式
    for i in range(len(metrics_data)):
        for j in range(len(metrics_data[0])):
            if i == 0:  # 表头
                table[(i, j)].set_facecolor('#4CAF50')
                table[(i, j)].set_text_props(weight='bold', color='white')
            else:
                table[(i, j)].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    
    plt.suptitle(title, fontsize=16, y=0.95)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"综合性能分析图已保存到: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()