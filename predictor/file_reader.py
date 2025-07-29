import csv
import pandas as pd
from pathlib import Path
from typing import List, Optional, Union


class ExpressionFileReader:
    """
    表达式文件读取器
    
    支持从TXT和CSV文件读取alpha表达式
    """
    
    @staticmethod
    def read_txt_file(file_path: str) -> List[str]:
        """
        从TXT文件读取表达式
        
        Args:
            file_path: TXT文件路径
            
        Returns:
            表达式列表
        """
        expressions = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # 跳过空行和注释行
                    if not line or line.startswith('#'):
                        continue
                    
                    expressions.append(line)
            
            return expressions
            
        except Exception as e:
            raise IOError(f"读取TXT文件失败 {file_path}: {e}")
    
    @staticmethod
    def read_csv_file(file_path: str, 
                      column: Union[str, int, None] = None,
                      has_header: bool = True) -> List[str]:
        """
        从CSV文件读取表达式
        
        Args:
            file_path: CSV文件路径
            column: 列名称或索引，None表示自动检测
            has_header: 是否有标题行
            
        Returns:
            表达式列表
        """
        try:
            # 读取CSV文件
            if has_header:
                df = pd.read_csv(file_path)
            else:
                df = pd.read_csv(file_path, header=None)
            
            if df.empty:
                return []
            
            # 确定要读取的列
            target_column = ExpressionFileReader._determine_target_column(
                df, column, has_header
            )
            
            # 提取表达式
            expressions = []
            for idx, value in enumerate(df[target_column]):
                if pd.isna(value):
                    continue
                
                expression = str(value).strip()
                if not expression:
                    continue
                
                expressions.append(expression)
            
            return expressions
            
        except Exception as e:
            raise IOError(f"读取CSV文件失败 {file_path}: {e}")
    
    @staticmethod
    def _determine_target_column(df: pd.DataFrame, 
                                column: Union[str, int, None],
                                has_header: bool) -> Union[str, int]:
        """
        确定目标列
        
        Args:
            df: DataFrame
            column: 指定的列
            has_header: 是否有标题行
            
        Returns:
            目标列名或索引
        """
        if column is not None:
            # 用户指定了列
            if isinstance(column, str):
                if column in df.columns:
                    return column
                else:
                    raise ValueError(f"列 '{column}' 不存在。可用列: {list(df.columns)}")
            
            elif isinstance(column, int):
                if 0 <= column < len(df.columns):
                    return df.columns[column] if has_header else column
                else:
                    raise ValueError(f"列索引 {column} 超出范围。列数: {len(df.columns)}")
        
        else:
            # 自动检测列
            if has_header:
                # 寻找包含 'expression', 'expr', 'formula' 等关键词的列
                keywords = ['expression', 'expr', 'formula', 'alpha']
                for col_name in df.columns:
                    if any(keyword in col_name.lower() for keyword in keywords):
                        return col_name
                
                # 如果没找到，使用第一列
                return df.columns[0]
            else:
                # 没有标题行，使用第一列
                return 0
    
    @staticmethod
    def auto_detect_file_type(file_path: str) -> str:
        """
        自动检测文件类型
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件类型 ('txt' 或 'csv')
        """
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.csv':
            return 'csv'
        elif file_extension in ['.txt', '.text']:
            return 'txt'
        else:
            # 尝试检测内容
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                    
                # 如果包含逗号，可能是CSV
                if ',' in first_line and len(first_line.split(',')) > 1:
                    return 'csv'
                else:
                    return 'txt'
                    
            except:
                # 默认当作TXT处理
                return 'txt'
    
    @staticmethod
    def read_expressions_from_file(file_path: str,
                                   file_type: Optional[str] = None,
                                   column: Union[str, int, None] = None,
                                   has_header: bool = True) -> List[str]:
        """
        从文件读取表达式（统一接口）
        
        Args:
            file_path: 文件路径
            file_type: 文件类型 ('txt', 'csv' 或 None自动检测)
            column: CSV列名称或索引
            has_header: CSV是否有标题行
            
        Returns:
            表达式列表
        """
        # 检查文件是否存在
        if not Path(file_path).exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 自动检测文件类型
        if file_type is None:
            file_type = ExpressionFileReader.auto_detect_file_type(file_path)
        
        # 根据文件类型读取
        if file_type == 'csv':
            return ExpressionFileReader.read_csv_file(file_path, column, has_header)
        elif file_type == 'txt':
            return ExpressionFileReader.read_txt_file(file_path)
        else:
            raise ValueError(f"不支持的文件类型: {file_type}")
    
    @staticmethod
    def write_results_to_csv(results: List[dict], output_path: str):
        """
        将预测结果写入CSV文件
        
        Args:
            results: 预测结果列表
            output_path: 输出文件路径
        """
        if not results:
            return
        
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                # 确定字段名
                fieldnames = ['expression', 'predicted_sharpe']
                
                # 检查是否有其他字段
                if results[0].get('parsed_expression'):
                    fieldnames.append('parsed_expression')
                if results[0].get('variable_mapping'):
                    fieldnames.append('variable_mapping')
                if any('error' in result for result in results):
                    fieldnames.append('error')
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in results:
                    # 处理变量映射的格式
                    row = result.copy()
                    if 'variable_mapping' in row and row['variable_mapping']:
                        row['variable_mapping'] = str(row['variable_mapping'])
                    
                    writer.writerow(row)
            
            print(f"结果已保存到: {output_path}")
            
        except Exception as e:
            raise IOError(f"写入CSV文件失败 {output_path}: {e}")


def main():
    """测试文件读取功能"""
    print("=== 文件读取器测试 ===")
    
    # 创建测试文件
    test_txt = "/tmp/test_expressions.txt"
    test_csv = "/tmp/test_expressions.csv"
    
    # 创建TXT测试文件
    with open(test_txt, 'w') as f:
        f.write("abs(div(x1,x2));2 TCFBaseClean 2 TCFBaseClean 5\n")
        f.write("sub(x1,x2);2 TCFBaseClean 1 TCFBaseClean 2\n")
        f.write("# 这是注释\n")
        f.write("ts_avg(x1,10);1 TCFBaseClean 5\n")
    
    # 创建CSV测试文件
    with open(test_csv, 'w') as f:
        f.write("name,expression,description\n")
        f.write("Alpha1,\"abs(div(x1,x2));2 TCFBaseClean 2 TCFBaseClean 5\",Test alpha 1\n")
        f.write("Alpha2,\"sub(x1,x2);2 TCFBaseClean 1 TCFBaseClean 2\",Test alpha 2\n")
    
    reader = ExpressionFileReader()
    
    # 测试TXT读取
    print("\n测试TXT文件读取:")
    txt_expressions = reader.read_expressions_from_file(test_txt)
    for i, expr in enumerate(txt_expressions, 1):
        print(f"  {i}. {expr}")
    
    # 测试CSV读取
    print("\n测试CSV文件读取:")
    csv_expressions = reader.read_expressions_from_file(test_csv, column='expression')
    for i, expr in enumerate(csv_expressions, 1):
        print(f"  {i}. {expr}")
    
    # 测试自动检测
    print("\n测试自动检测:")
    auto_txt = reader.read_expressions_from_file(test_txt)
    auto_csv = reader.read_expressions_from_file(test_csv, column='expression')
    print(f"  TXT自动检测: {len(auto_txt)} 个表达式")
    print(f"  CSV自动检测: {len(auto_csv)} 个表达式")
    
    # 清理测试文件
    Path(test_txt).unlink()
    Path(test_csv).unlink()


if __name__ == "__main__":
    main()