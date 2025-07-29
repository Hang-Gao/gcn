#!/usr/bin/env python3
"""
Alpha表达式Sharpe值预测器 - 命令行界面

使用训练好的GCN模型预测alpha表达式的Sharpe值
"""

import sys
import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from predictor.alpha_predictor import AlphaPredictor
from predictor.file_reader import ExpressionFileReader


def setup_logging(verbose: bool = False):
    """设置日志"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def format_output(results: List[Dict[str, Any]], 
                  output_format: str = 'detailed') -> str:
    """
    格式化输出结果
    
    Args:
        results: 预测结果列表
        output_format: 输出格式 ('simple', 'detailed', 'json')
        
    Returns:
        格式化的输出字符串
    """
    if output_format == 'json':
        return json.dumps(results, indent=2, ensure_ascii=False)
    
    output_lines = []
    
    for i, result in enumerate(results, 1):
        if 'error' in result:
            if output_format == 'simple':
                output_lines.append(f"ERROR")
            else:
                output_lines.append(f"{i}. 表达式: {result['expression']}")
                output_lines.append(f"   错误: {result['error']}")
        else:
            sharpe = result['predicted_sharpe']
            if output_format == 'simple':
                output_lines.append(f"{sharpe:.6f}")
            else:
                output_lines.append(f"{i}. 表达式: {result['expression']}")
                output_lines.append(f"   预测Sharpe值: {sharpe:.6f}")
                if 'variable_mapping' in result:
                    output_lines.append(f"   变量映射: {result['variable_mapping']}")
    
    return '\n'.join(output_lines)


def interactive_mode(predictor: AlphaPredictor):
    """交互式模式"""
    print("=== Alpha表达式预测器 - 交互模式 ===")
    print("输入alpha表达式进行预测，输入 'quit' 或 'exit' 退出")
    print("格式: 表达式;num 数据1名称 数据1编号 数据2名称 数据2编号...")
    print("示例: abs(div(x1,x2));2 TCFBaseClean 2 TCFBaseClean 5")
    print()
    
    history = []
    
    while True:
        try:
            expression = input("请输入表达式: ").strip()
            
            if expression.lower() in ['quit', 'exit', 'q']:
                break
            
            if not expression:
                continue
            
            # 预测
            result = predictor.predict_single(expression)
            history.append(result)
            
            # 输出结果
            if 'error' in result:
                print(f"错误: {result['error']}")
            else:
                print(f"预测Sharpe值: {result['predicted_sharpe']:.6f}")
                if 'variable_mapping' in result:
                    print(f"变量映射: {result['variable_mapping']}")
            
            print()
            
        except KeyboardInterrupt:
            print("\n\n用户中断，退出交互模式")
            break
        except Exception as e:
            print(f"发生错误: {e}")
    
    print(f"\n本次会话共预测了 {len(history)} 个表达式")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Alpha表达式Sharpe值预测器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 预测单个表达式
  python predict.py --expression "abs(div(x1,x2));2 TCFBaseClean 2 TCFBaseClean 5"
  
  # 预测多个表达式
  python predict.py --expressions "abs(x1);1 TCFBaseClean 5" "sub(x1,x2);2 TCFBaseClean 1 TCFBaseClean 2"
  
  # 从文件批量预测
  python predict.py --file expressions.txt
  python predict.py --file expressions.csv --column expression
  
  # 交互式模式
  python predict.py --interactive
  
  # 指定模型和输出格式
  python predict.py --model latest --expression "abs(x1);1 TCFBaseClean 5" --format json
  
  # 保存结果到文件
  python predict.py --file input.csv --output results.csv
        """
    )
    
    # 输入选项组
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--expression', '-e', 
                           help='单个alpha表达式')
    input_group.add_argument('--expressions', '-E', nargs='+',
                           help='多个alpha表达式')
    input_group.add_argument('--file', '-f',
                           help='包含表达式的文件路径 (支持.txt和.csv)')
    input_group.add_argument('--interactive', '-i', action='store_true',
                           help='进入交互式模式')
    input_group.add_argument('--list-models', action='store_true',
                           help='列出所有可用模型')
    
    # 文件选项
    parser.add_argument('--column', '-c', 
                       help='CSV文件中的列名或列索引 (默认自动检测)')
    parser.add_argument('--no-header', action='store_true',
                       help='CSV文件没有标题行')
    
    # 模型选项
    parser.add_argument('--model', '-m',
                       help='模型选择 (latest, 模型文件路径, 或模型索引)')
    
    # 输出选项
    parser.add_argument('--format', choices=['simple', 'detailed', 'json'],
                       default='detailed',
                       help='输出格式 (默认: detailed)')
    parser.add_argument('--output', '-o',
                       help='输出文件路径 (仅支持CSV格式)')
    
    # 其他选项
    parser.add_argument('--device',
                       choices=['cpu', 'cuda', 'mps'],
                       help='计算设备 (默认自动选择)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='显示详细日志')
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.verbose)
    
    try:
        # 列出模型
        if args.list_models:
            from predictor.model_manager import ModelManager
            manager = ModelManager()
            models = manager.list_available_models()
            
            if models:
                print("可用模型:")
                for desc in models:
                    print(f"  {desc}")
            else:
                print("未找到任何模型文件")
            return
        
        # 初始化预测器
        print("初始化预测器...")
        predictor = AlphaPredictor(model_path=args.model, device=args.device)
        
        # 显示模型信息
        if args.verbose:
            model_info = predictor.get_model_info()
            print(f"使用模型: {Path(model_info['model_path']).name}")
            print(f"计算设备: {model_info['device']}")
            print()
        
        # 处理不同的输入模式
        results = []
        
        if args.interactive:
            # 交互式模式
            interactive_mode(predictor)
            return
        
        elif args.expression:
            # 单个表达式
            result = predictor.predict_single(args.expression)
            results = [result]
        
        elif args.expressions:
            # 多个表达式
            results = predictor.predict_batch(args.expressions)
        
        elif args.file:
            # 从文件读取
            print(f"从文件读取表达式: {args.file}")
            
            # 处理列参数
            column = args.column
            if column and column.isdigit():
                column = int(column)
            
            expressions = ExpressionFileReader.read_expressions_from_file(
                args.file, 
                column=column,
                has_header=not args.no_header
            )
            
            print(f"读取到 {len(expressions)} 个表达式")
            
            if expressions:
                print("开始批量预测...")
                results = predictor.predict_batch(expressions)
            else:
                print("文件中没有找到有效的表达式")
                return
        
        # 输出结果
        if results:
            output_text = format_output(results, args.format)
            print("\n=== 预测结果 ===")
            print(output_text)
            
            # 保存到文件
            if args.output:
                ExpressionFileReader.write_results_to_csv(results, args.output)
            
            # 统计信息
            success_count = sum(1 for r in results if 'error' not in r)
            error_count = len(results) - success_count
            
            if args.format != 'simple':
                print(f"\n=== 统计信息 ===")
                print(f"总计: {len(results)} 个表达式")
                print(f"成功: {success_count} 个")
                if error_count > 0:
                    print(f"失败: {error_count} 个")
    
    except KeyboardInterrupt:
        print("\n用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"错误: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()