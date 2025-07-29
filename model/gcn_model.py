import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.data import Batch
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class AlphaGCN(nn.Module):
    """
    Alpha表达式GCN模型
    
    使用图卷积网络预测Alpha表达式的Sharpe值
    """
    
    def __init__(self, 
                 node_feature_dim: int,
                 edge_feature_dim: int = 2,
                 graph_feature_dim: int = 7,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 dropout: float = 0.2,
                 pool_method: str = 'mean',
                 use_edge_features: bool = True,
                 use_graph_features: bool = True,
                 use_attention: bool = False):
        """
        初始化GCN模型
        
        Args:
            node_feature_dim: 节点特征维度
            edge_feature_dim: 边特征维度
            graph_feature_dim: 图级特征维度
            hidden_dim: 隐藏层维度
            num_layers: GCN层数
            dropout: dropout率
            pool_method: 池化方法 ('mean', 'add', 'max')
            use_edge_features: 是否使用边特征
            use_graph_features: 是否使用图级特征
            use_attention: 是否使用注意力机制(GAT)
        """
        super(AlphaGCN, self).__init__()
        
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.graph_feature_dim = graph_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.pool_method = pool_method
        self.use_edge_features = use_edge_features
        self.use_graph_features = use_graph_features
        self.use_attention = use_attention
        
        # 节点特征预处理
        self.node_embedding = nn.Linear(node_feature_dim, hidden_dim)
        
        # 图卷积层
        self.conv_layers = nn.ModuleList()
        
        if use_attention:
            # 使用图注意力网络
            for i in range(num_layers):
                if i == 0:
                    self.conv_layers.append(
                        GATConv(hidden_dim, hidden_dim // 4, heads=4, dropout=dropout)
                    )
                else:
                    self.conv_layers.append(
                        GATConv(hidden_dim, hidden_dim // 4, heads=4, dropout=dropout)
                    )
        else:
            # 使用标准GCN
            for i in range(num_layers):
                if i == 0:
                    self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
                else:
                    self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
        
        # 批归一化层
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        
        # 边特征处理（如果使用）
        if use_edge_features:
            self.edge_embedding = nn.Linear(edge_feature_dim, hidden_dim // 4)
        
        # 池化层选择
        if pool_method == 'mean':
            self.pool = global_mean_pool
        elif pool_method == 'add':
            self.pool = global_add_pool
        elif pool_method == 'max':
            self.pool = global_max_pool
        else:
            raise ValueError(f"不支持的池化方法: {pool_method}")
        
        # 图级特征融合
        final_dim = hidden_dim
        if use_graph_features:
            self.graph_feature_mlp = nn.Sequential(
                nn.Linear(graph_feature_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, hidden_dim // 4)
            )
            final_dim += hidden_dim // 4
        
        # 最终预测层
        self.predictor = nn.Sequential(
            nn.Linear(final_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout // 2),
            nn.Linear(hidden_dim // 4, 1)  # 单个Sharpe值输出
        )
        
        # 初始化权重
        self._init_weights()
        
        logger.info(f"AlphaGCN模型初始化完成:")
        logger.info(f"  节点特征维度: {node_feature_dim}")
        logger.info(f"  隐藏层维度: {hidden_dim}")
        logger.info(f"  GCN层数: {num_layers}")
        logger.info(f"  池化方法: {pool_method}")
        logger.info(f"  使用边特征: {use_edge_features}")
        logger.info(f"  使用图特征: {use_graph_features}")
        logger.info(f"  使用注意力: {use_attention}")
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, batch: Batch) -> torch.Tensor:
        """
        前向传播
        
        Args:
            batch: 批次图数据
            
        Returns:
            预测的Sharpe值 (batch_size, 1)
        """
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
        
        # 1. 节点特征嵌入
        x = self.node_embedding(x)
        x = F.relu(x)
        
        # 2. 图卷积层
        for i, conv in enumerate(self.conv_layers):
            x_residual = x
            
            # 图卷积
            if self.use_attention:
                x = conv(x, edge_index)
            else:
                x = conv(x, edge_index)
            
            # 批归一化
            x = self.batch_norms[i](x)
            
            # 激活函数
            x = F.relu(x)
            
            # 残差连接（从第二层开始）
            if i > 0:
                x = x + x_residual
            
            # Dropout
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 3. 图级池化
        graph_embedding = self.pool(x, batch_idx)  # (batch_size, hidden_dim)
        
        # 4. 融合图级特征（如果使用）
        if self.use_graph_features and hasattr(batch, 'graph_features'):
            # batch.graph_features是连接后的向量，需要reshape为(batch_size, graph_feature_dim)
            batch_size = graph_embedding.size(0)
            single_graph_feature_dim = self.graph_feature_dim
            
            # 将连接的特征重新reshape
            graph_features_reshaped = batch.graph_features.view(batch_size, single_graph_feature_dim)
            graph_features = self.graph_feature_mlp(graph_features_reshaped)
            graph_embedding = torch.cat([graph_embedding, graph_features], dim=1)
        
        # 5. 最终预测
        output = self.predictor(graph_embedding)
        
        return output
    
    def get_embeddings(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """
        获取中间嵌入表示
        
        Args:
            batch: 批次图数据
            
        Returns:
            各层嵌入字典
        """
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
        embeddings = {}
        
        # 节点特征嵌入
        x = self.node_embedding(x)
        x = F.relu(x)
        embeddings['node_embedding'] = x
        
        # 图卷积层嵌入
        for i, conv in enumerate(self.conv_layers):
            x_residual = x
            
            if self.use_attention:
                x = conv(x, edge_index)
            else:
                x = conv(x, edge_index)
            
            x = self.batch_norms[i](x)
            x = F.relu(x)
            
            if i > 0:
                x = x + x_residual
            
            x = F.dropout(x, p=self.dropout, training=self.training)
            embeddings[f'conv_layer_{i}'] = x
        
        # 图级嵌入
        graph_embedding = self.pool(x, batch_idx)
        embeddings['graph_embedding'] = graph_embedding
        
        # 融合图级特征
        if self.use_graph_features and hasattr(batch, 'graph_features'):
            batch_size = graph_embedding.size(0)
            single_graph_feature_dim = self.graph_feature_dim
            graph_features_reshaped = batch.graph_features.view(batch_size, single_graph_feature_dim)
            graph_features = self.graph_feature_mlp(graph_features_reshaped)
            final_embedding = torch.cat([graph_embedding, graph_features], dim=1)
            embeddings['final_embedding'] = final_embedding
        else:
            embeddings['final_embedding'] = graph_embedding
        
        return embeddings


class AlphaGCNEnsemble(nn.Module):
    """
    Alpha GCN集成模型
    
    使用多个不同配置的GCN模型进行集成预测
    """
    
    def __init__(self, models: List[AlphaGCN], weights: Optional[List[float]] = None):
        """
        初始化集成模型
        
        Args:
            models: GCN模型列表
            weights: 模型权重（如果为None，则使用平均权重）
        """
        super(AlphaGCNEnsemble, self).__init__()
        
        self.models = nn.ModuleList(models)
        
        if weights is None:
            self.weights = torch.ones(len(models)) / len(models)
        else:
            self.weights = torch.tensor(weights, dtype=torch.float32)
        
        logger.info(f"GCN集成模型初始化完成，包含 {len(models)} 个子模型")
    
    def forward(self, batch: Batch) -> torch.Tensor:
        """
        前向传播
        
        Args:
            batch: 批次图数据
            
        Returns:
            集成预测结果
        """
        predictions = []
        
        for model in self.models:
            pred = model(batch)
            predictions.append(pred)
        
        # 加权平均
        predictions = torch.stack(predictions, dim=2)  # (batch_size, 1, num_models)
        weights = self.weights.view(1, 1, -1).to(predictions.device)
        
        ensemble_pred = torch.sum(predictions * weights, dim=2)
        
        return ensemble_pred


def create_alpha_gcn_model(feature_dims: Dict[str, int], 
                          config: Dict[str, Any]) -> AlphaGCN:
    """
    创建Alpha GCN模型的工厂函数
    
    Args:
        feature_dims: 特征维度字典
        config: 模型配置
        
    Returns:
        配置好的AlphaGCN模型
    """
    model = AlphaGCN(
        node_feature_dim=feature_dims['node_feature_dim'],
        edge_feature_dim=feature_dims.get('edge_feature_dim', 2),
        graph_feature_dim=feature_dims.get('graph_feature_dim', 7),
        hidden_dim=config.get('hidden_dim', 128),
        num_layers=config.get('num_layers', 3),
        dropout=config.get('dropout', 0.2),
        pool_method=config.get('pool_method', 'mean'),
        use_edge_features=config.get('use_edge_features', True),
        use_graph_features=config.get('use_graph_features', True),
        use_attention=config.get('use_attention', False)
    )
    
    return model


def create_ensemble_models(feature_dims: Dict[str, int],
                          ensemble_configs: List[Dict[str, Any]]) -> AlphaGCNEnsemble:
    """
    创建集成模型
    
    Args:
        feature_dims: 特征维度字典
        ensemble_configs: 集成模型配置列表
        
    Returns:
        集成模型
    """
    models = []
    
    for config in ensemble_configs:
        model = create_alpha_gcn_model(feature_dims, config)
        models.append(model)
    
    ensemble = AlphaGCNEnsemble(models)
    return ensemble


# 预定义的模型配置
DEFAULT_CONFIG = {
    'hidden_dim': 128,
    'num_layers': 3,
    'dropout': 0.2,
    'pool_method': 'mean',
    'use_edge_features': True,
    'use_graph_features': True,
    'use_attention': False
}

LIGHTWEIGHT_CONFIG = {
    'hidden_dim': 64,
    'num_layers': 2,
    'dropout': 0.1,
    'pool_method': 'mean',
    'use_edge_features': False,
    'use_graph_features': True,
    'use_attention': False
}

ATTENTION_CONFIG = {
    'hidden_dim': 128,
    'num_layers': 3,
    'dropout': 0.3,
    'pool_method': 'mean',
    'use_edge_features': True,
    'use_graph_features': True,
    'use_attention': True
}

ENSEMBLE_CONFIGS = [
    DEFAULT_CONFIG,
    LIGHTWEIGHT_CONFIG,
    ATTENTION_CONFIG
]