import os
import glob
from typing import Optional, List, Tuple
from pathlib import Path


class ModelManager:
    """模型管理器 - 负责模型文件的发现、选择和验证"""
    
    def __init__(self, models_dir: str = "results/model_checkpoints"):
        """
        初始化模型管理器
        
        Args:
            models_dir: 模型文件目录路径
        """
        self.models_dir = Path(models_dir)
        if not self.models_dir.is_absolute():
            # 相对于项目根目录
            project_root = Path(__file__).parent.parent
            self.models_dir = project_root / models_dir
    
    def get_all_models(self) -> List[Tuple[str, float]]:
        """
        获取所有可用的模型文件
        
        Returns:
            [(模型路径, 修改时间戳), ...] 按时间倒序排列
        """
        if not self.models_dir.exists():
            return []
        
        pattern = str(self.models_dir / "*.pth")
        model_files = glob.glob(pattern)
        
        # 获取文件修改时间并排序
        models_with_time = []
        for model_file in model_files:
            try:
                mtime = os.path.getmtime(model_file)
                models_with_time.append((model_file, mtime))
            except OSError:
                continue
        
        # 按修改时间倒序排列（最新的在前）
        models_with_time.sort(key=lambda x: x[1], reverse=True)
        
        return models_with_time
    
    def get_latest_model(self) -> Optional[str]:
        """
        获取最新的模型文件路径
        
        Returns:
            最新模型文件路径，如果没有找到返回None
        """
        models = self.get_all_models()
        if models:
            return models[0][0]
        return None
    
    def list_available_models(self) -> List[str]:
        """
        列出所有可用模型的简短描述
        
        Returns:
            模型描述列表
        """
        models = self.get_all_models()
        descriptions = []
        
        for i, (model_path, mtime) in enumerate(models):
            model_name = Path(model_path).name
            # 从文件名提取时间戳
            if "alpha_gcn_model_" in model_name:
                timestamp = model_name.replace("alpha_gcn_model_", "").replace(".pth", "")
                desc = f"{i+1}. {model_name} (训练时间: {timestamp})"
            else:
                desc = f"{i+1}. {model_name}"
            
            descriptions.append(desc)
        
        return descriptions
    
    def validate_model_file(self, model_path: str) -> bool:
        """
        验证模型文件是否存在且可读
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            是否有效
        """
        try:
            path = Path(model_path)
            return path.exists() and path.is_file() and path.suffix == '.pth'
        except:
            return False
    
    def select_model(self, model_specification: Optional[str] = None) -> Optional[str]:
        """
        智能选择模型
        
        Args:
            model_specification: 模型规格说明
                - None: 自动选择最新模型
                - "latest": 明确选择最新模型  
                - 文件路径: 指定具体模型文件
                - 数字: 选择第N个模型（从列表中）
                
        Returns:
            选中的模型文件路径
        """
        if model_specification is None or model_specification == "latest":
            return self.get_latest_model()
        
        # 如果是文件路径
        if "/" in model_specification or "\\" in model_specification or model_specification.endswith(".pth"):
            if self.validate_model_file(model_specification):
                return model_specification
            else:
                raise FileNotFoundError(f"模型文件不存在或无效: {model_specification}")
        
        # 如果是数字索引
        try:
            index = int(model_specification) - 1  # 用户输入1-based，转换为0-based
            models = self.get_all_models()
            if 0 <= index < len(models):
                return models[index][0]
            else:
                raise ValueError(f"模型索引超出范围: {model_specification} (可用范围: 1-{len(models)})")
        except ValueError as e:
            if "invalid literal" in str(e):
                raise ValueError(f"无效的模型规格说明: {model_specification}")
            else:
                raise e


def main():
    """测试模型管理器功能"""
    manager = ModelManager()
    
    print("=== 模型管理器测试 ===")
    
    # 列出所有模型
    models = manager.list_available_models()
    if models:
        print("\n可用模型:")
        for desc in models:
            print(f"  {desc}")
        
        # 获取最新模型
        latest = manager.get_latest_model()
        print(f"\n最新模型: {latest}")
    else:
        print("\n未找到任何模型文件")
    
    # 测试模型选择
    test_cases = [None, "latest", "1", "2"]
    for case in test_cases:
        try:
            selected = manager.select_model(case)
            print(f"\n选择规格 '{case}' -> {selected}")
        except Exception as e:
            print(f"\n选择规格 '{case}' -> 错误: {e}")


if __name__ == "__main__":
    main()