"""
评估指标计算模块
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    指标计算器类
    """
    
    @staticmethod
    def calculate_basic_metrics(predictions: List[float], targets: List[float]) -> Dict[str, float]:
        """
        计算基础回归指标
        
        Args:
            predictions: 预测值列表
            targets: 真实值列表
            
        Returns:
            基础指标字典
        """
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        rmse = np.sqrt(mse)
        
        # 相关系数
        correlation = np.corrcoef(predictions, targets)[0, 1] if len(predictions) > 1 else 0.0
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'rmse': rmse,
            'correlation': correlation
        }
    
    @staticmethod
    def calculate_error_statistics(predictions: List[float], targets: List[float]) -> Dict[str, Any]:
        """
        计算误差统计信息
        
        Args:
            predictions: 预测值列表
            targets: 真实值列表
            
        Returns:
            误差统计字典
        """
        predictions = np.array(predictions)
        targets = np.array(targets)
        errors = predictions - targets
        abs_errors = np.abs(errors)
        
        # 基础统计
        error_stats = {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'max_error': np.max(abs_errors),
            'min_error': np.min(abs_errors),
            'median_error': np.median(errors),
            'median_abs_error': np.median(abs_errors)
        }
        
        # 分位数
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        error_stats['error_percentiles'] = {}
        for p in percentiles:
            error_stats['error_percentiles'][f'{p}%'] = np.percentile(abs_errors, p)
        
        # 相对误差（避免除零）
        relative_errors = abs_errors / (np.abs(targets) + 1e-8)
        error_stats['mean_relative_error'] = np.mean(relative_errors)
        error_stats['median_relative_error'] = np.median(relative_errors)
        
        return error_stats
    
    @staticmethod
    def calculate_distribution_metrics(predictions: List[float], targets: List[float]) -> Dict[str, float]:
        """
        计算分布相关指标
        
        Args:
            predictions: 预测值列表
            targets: 真实值列表
            
        Returns:
            分布指标字典
        """
        predictions = np.array(predictions)
        targets = np.array(targets)
        errors = predictions - targets
        
        # 正态性检验 (Shapiro-Wilk test)
        if len(errors) >= 3:
            shapiro_stat, shapiro_p = stats.shapiro(errors)
        else:
            shapiro_stat, shapiro_p = 0.0, 1.0
        
        # 偏度和峰度
        skewness = stats.skew(errors)
        kurtosis = stats.kurtosis(errors)
        
        # Kolmogorov-Smirnov 正态性检验
        if len(errors) >= 3:
            ks_stat, ks_p = stats.kstest(errors, 'norm')
        else:
            ks_stat, ks_p = 0.0, 1.0
        
        return {
            'error_skewness': skewness,
            'error_kurtosis': kurtosis,
            'shapiro_stat': shapiro_stat,
            'shapiro_pvalue': shapiro_p,
            'ks_stat': ks_stat,
            'ks_pvalue': ks_p,
            'is_normal_shapiro': shapiro_p > 0.05,
            'is_normal_ks': ks_p > 0.05
        }
    
    @staticmethod
    def calculate_prediction_quality(predictions: List[float], targets: List[float]) -> Dict[str, Any]:
        """
        计算预测质量指标
        
        Args:
            predictions: 预测值列表
            targets: 真实值列表
            
        Returns:
            预测质量指标字典
        """
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # 预测准确性分级
        abs_errors = np.abs(predictions - targets)
        target_range = np.max(targets) - np.min(targets)
        
        # 定义准确性阈值（基于目标值范围的百分比）
        thresholds = {
            'excellent': 0.01 * target_range,  # 1%
            'good': 0.03 * target_range,       # 3%
            'fair': 0.05 * target_range,       # 5%
            'poor': 0.10 * target_range        # 10%
        }
        
        # 计算各准确性级别的样本比例
        quality_counts = {
            'excellent': np.sum(abs_errors <= thresholds['excellent']),
            'good': np.sum((abs_errors > thresholds['excellent']) & (abs_errors <= thresholds['good'])),
            'fair': np.sum((abs_errors > thresholds['good']) & (abs_errors <= thresholds['fair'])),
            'poor': np.sum((abs_errors > thresholds['fair']) & (abs_errors <= thresholds['poor'])),
            'very_poor': np.sum(abs_errors > thresholds['poor'])
        }
        
        total_samples = len(predictions)
        quality_ratios = {k: v / total_samples for k, v in quality_counts.items()}
        
        return {
            'quality_counts': quality_counts,
            'quality_ratios': quality_ratios,
            'thresholds': thresholds,
            'target_range': target_range,
            'total_samples': total_samples
        }


def calculate_comprehensive_metrics(predictions: List[float], 
                                  targets: List[float],
                                  detailed: bool = True) -> Dict[str, Any]:
    """
    计算综合评估指标
    
    Args:
        predictions: 预测值列表
        targets: 真实值列表
        detailed: 是否计算详细指标
        
    Returns:
        综合指标字典
    """
    calculator = MetricsCalculator()
    
    # 基础指标
    basic_metrics = calculator.calculate_basic_metrics(predictions, targets)
    result = {'basic_metrics': basic_metrics}
    
    if detailed:
        # 误差统计
        error_stats = calculator.calculate_error_statistics(predictions, targets)
        result['error_statistics'] = error_stats
        
        # 分布指标
        distribution_metrics = calculator.calculate_distribution_metrics(predictions, targets)
        result['distribution_metrics'] = distribution_metrics
        
        # 预测质量
        quality_metrics = calculator.calculate_prediction_quality(predictions, targets)
        result['quality_metrics'] = quality_metrics
    
    # 添加元信息
    result['meta'] = {
        'num_samples': len(predictions),
        'target_range': np.max(targets) - np.min(targets),
        'target_mean': np.mean(targets),
        'target_std': np.std(targets),
        'prediction_mean': np.mean(predictions),
        'prediction_std': np.std(predictions)
    }
    
    logger.info(f"指标计算完成，样本数: {len(predictions)}")
    logger.info(f"基础指标 - MSE: {basic_metrics['mse']:.6f}, R²: {basic_metrics['r2']:.4f}")
    
    return result


def print_metrics_summary(metrics: Dict[str, Any], title: str = "模型评估结果") -> None:
    """
    打印指标摘要
    
    Args:
        metrics: 指标字典
        title: 标题
    """
    print(f"\n{'='*50}")
    print(f"{title:^50}")
    print(f"{'='*50}")
    
    # 基础指标
    if 'basic_metrics' in metrics:
        basic = metrics['basic_metrics']
        print(f"\n📊 基础指标:")
        print(f"  MSE (均方误差):        {basic['mse']:.6f}")
        print(f"  MAE (平均绝对误差):     {basic['mae']:.6f}")
        print(f"  RMSE (均方根误差):      {basic['rmse']:.6f}")
        print(f"  R² (决定系数):         {basic['r2']:.4f}")
        print(f"  相关系数:              {basic['correlation']:.4f}")
    
    # 元信息
    if 'meta' in metrics:
        meta = metrics['meta']
        print(f"\n📈 数据概览:")
        print(f"  样本数量:              {meta['num_samples']}")
        print(f"  目标值范围:            {meta['target_range']:.6f}")
        print(f"  目标值均值:            {meta['target_mean']:.6f}")
        print(f"  目标值标准差:          {meta['target_std']:.6f}")
    
    # 误差统计
    if 'error_statistics' in metrics:
        error = metrics['error_statistics']
        print(f"\n🎯 误差统计:")
        print(f"  平均误差:              {error['mean_error']:.6f}")
        print(f"  误差标准差:            {error['std_error']:.6f}")
        print(f"  最大绝对误差:          {error['max_error']:.6f}")
        print(f"  中位数绝对误差:        {error['median_abs_error']:.6f}")
        print(f"  平均相对误差:          {error['mean_relative_error']:.4f}")
    
    # 预测质量
    if 'quality_metrics' in metrics:
        quality = metrics['quality_metrics']
        ratios = quality['quality_ratios']
        print(f"\n⭐ 预测质量分布:")
        print(f"  优秀 (误差<1%):        {ratios['excellent']:.1%}")
        print(f"  良好 (误差1-3%):       {ratios['good']:.1%}")
        print(f"  一般 (误差3-5%):       {ratios['fair']:.1%}")
        print(f"  较差 (误差5-10%):      {ratios['poor']:.1%}")
        print(f"  很差 (误差>10%):       {ratios['very_poor']:.1%}")
    
    # 分布特性
    if 'distribution_metrics' in metrics:
        dist = metrics['distribution_metrics']
        print(f"\n📊 误差分布特性:")
        print(f"  偏度:                 {dist['error_skewness']:.4f}")
        print(f"  峰度:                 {dist['error_kurtosis']:.4f}")
        print(f"  正态性(Shapiro):       {'✓' if dist['is_normal_shapiro'] else '✗'} (p={dist['shapiro_pvalue']:.4f})")
        print(f"  正态性(K-S):          {'✓' if dist['is_normal_ks'] else '✗'} (p={dist['ks_pvalue']:.4f})")
    
    print(f"\n{'='*50}")


def compare_models(model_results: Dict[str, Dict[str, Any]], 
                  metric_name: str = 'r2') -> None:
    """
    比较多个模型的性能
    
    Args:
        model_results: 模型结果字典，格式为 {model_name: metrics_dict}
        metric_name: 用于比较的主要指标名称
    """
    print(f"\n{'='*60}")
    print(f"{'模型性能比较':^60}")
    print(f"{'='*60}")
    
    # 提取比较指标
    comparison_data = []
    for model_name, metrics in model_results.items():
        if 'basic_metrics' in metrics:
            basic = metrics['basic_metrics']
            comparison_data.append({
                'model': model_name,
                'mse': basic['mse'],
                'mae': basic['mae'], 
                'r2': basic['r2'],
                'correlation': basic['correlation']
            })
    
    if not comparison_data:
        print("没有有效的模型结果用于比较")
        return
    
    # 按指定指标排序（R²越大越好，其他指标越小越好）
    reverse_sort = metric_name in ['r2', 'correlation']
    comparison_data.sort(key=lambda x: x[metric_name], reverse=reverse_sort)
    
    # 打印比较表格
    print(f"{'排名':<4} {'模型名称':<20} {'MSE':<12} {'MAE':<12} {'R²':<8} {'相关性':<8}")
    print("-" * 60)
    
    for i, data in enumerate(comparison_data, 1):
        print(f"{i:<4} {data['model']:<20} {data['mse']:<12.6f} {data['mae']:<12.6f} "
              f"{data['r2']:<8.4f} {data['correlation']:<8.4f}")
    
    # 最佳模型
    best_model = comparison_data[0]
    print(f"\n🏆 最佳模型: {best_model['model']}")
    print(f"   {metric_name.upper()}: {best_model[metric_name]:.6f}")
    
    print(f"\n{'='*60}")


def generate_metrics_report(metrics: Dict[str, Any], 
                          save_path: str = None) -> str:
    """
    生成指标报告
    
    Args:
        metrics: 指标字典
        save_path: 报告保存路径
        
    Returns:
        报告文本
    """
    from datetime import datetime
    
    report_lines = []
    report_lines.append("# Alpha GCN 模型评估报告")
    report_lines.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("\n" + "="*50)
    
    # 基础指标
    if 'basic_metrics' in metrics:
        basic = metrics['basic_metrics']
        report_lines.append("\n## 基础性能指标")
        report_lines.append(f"- **MSE (均方误差)**: {basic['mse']:.6f}")
        report_lines.append(f"- **MAE (平均绝对误差)**: {basic['mae']:.6f}")
        report_lines.append(f"- **RMSE (均方根误差)**: {basic['rmse']:.6f}")
        report_lines.append(f"- **R² (决定系数)**: {basic['r2']:.4f}")
        report_lines.append(f"- **Pearson相关系数**: {basic['correlation']:.4f}")
    
    # 数据概览
    if 'meta' in metrics:
        meta = metrics['meta']
        report_lines.append("\n## 数据概览")
        report_lines.append(f"- **测试样本数**: {meta['num_samples']}")
        report_lines.append(f"- **目标值范围**: {meta['target_range']:.6f}")
        report_lines.append(f"- **目标值均值**: {meta['target_mean']:.6f} ± {meta['target_std']:.6f}")
        report_lines.append(f"- **预测值均值**: {meta['prediction_mean']:.6f} ± {meta['prediction_std']:.6f}")
    
    # 误差分析
    if 'error_statistics' in metrics:
        error = metrics['error_statistics']
        report_lines.append("\n## 误差分析")
        report_lines.append(f"- **平均误差**: {error['mean_error']:.6f}")
        report_lines.append(f"- **误差标准差**: {error['std_error']:.6f}")
        report_lines.append(f"- **最大绝对误差**: {error['max_error']:.6f}")
        report_lines.append(f"- **平均相对误差**: {error['mean_relative_error']:.4f}")
        
        report_lines.append("\n### 误差分位数")
        for percentile, value in error['error_percentiles'].items():
            report_lines.append(f"- **{percentile}分位数**: {value:.6f}")
    
    # 预测质量
    if 'quality_metrics' in metrics:
        quality = metrics['quality_metrics']
        ratios = quality['quality_ratios']
        report_lines.append("\n## 预测质量分布")
        report_lines.append(f"- **优秀预测** (误差<1%): {ratios['excellent']:.1%}")
        report_lines.append(f"- **良好预测** (误差1-3%): {ratios['good']:.1%}")
        report_lines.append(f"- **一般预测** (误差3-5%): {ratios['fair']:.1%}")
        report_lines.append(f"- **较差预测** (误差5-10%): {ratios['poor']:.1%}")
        report_lines.append(f"- **很差预测** (误差>10%): {ratios['very_poor']:.1%}")
    
    # 分布特性
    if 'distribution_metrics' in metrics:
        dist = metrics['distribution_metrics']
        report_lines.append("\n## 误差分布特性")
        report_lines.append(f"- **偏度**: {dist['error_skewness']:.4f}")
        report_lines.append(f"- **峰度**: {dist['error_kurtosis']:.4f}")
        report_lines.append(f"- **正态性检验(Shapiro-Wilk)**: p={dist['shapiro_pvalue']:.4f} ({'通过' if dist['is_normal_shapiro'] else '不通过'})")
        report_lines.append(f"- **正态性检验(Kolmogorov-Smirnov)**: p={dist['ks_pvalue']:.4f} ({'通过' if dist['is_normal_ks'] else '不通过'})")
    
    report_text = "\n".join(report_lines)
    
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        logger.info(f"评估报告已保存到: {save_path}")
    
    return report_text