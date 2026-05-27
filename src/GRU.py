import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class CNNFeatureExtractor(nn.Module):
    
    def __init__(self, in_channels: int = 144, hidden_channels: int = 64) -> None:
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.residual_proj = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        self._init_weights()

    def _init_weights(self) -> None:
        
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = x.permute(0, 2, 1)
        residual = self.residual_proj(x)
        out = F.gelu(self.bn1(self.conv1(x)))
        out = F.gelu(self.bn2(self.conv2(out)))
        out = out + residual
        out = out.permute(0, 2, 1)
        return out

class GRUEncoder(nn.Module):
    
    def __init__(self, input_size: int = 64, hidden_size: int = 128) -> None:
        super().__init__()
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        out = self.layer_norm(out)
        return out

class TemporalAttention(nn.Module):
    
    def __init__(self, hidden_size: int = 128) -> None:
        super().__init__()
        
        self.attention_proj = nn.Linear(hidden_size, 1, bias=True)
        nn.init.xavier_uniform_(self.attention_proj.weight)
        nn.init.zeros_(self.attention_proj.bias)

    def forward(self, gru_out: torch.Tensor) -> torch.Tensor:
        
        scores = self.attention_proj(gru_out)
        weights = F.softmax(scores, dim=1)
        return (weights*gru_out).sum(dim=1)

class ClassificationHead(nn.Module):
    
    def __init__(self, hidden_size: int = 128, num_classes: int = 3) -> None:
        super().__init__()
        
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(64, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        
        for module in [self.fc1, self.fc2]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        out = self.relu(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        return out

class LOBModel(nn.Module):
    
    def __init__(self, in_channels: int = 144, cnn_channels: int = 64,\
                gru_hidden: int = 128, num_classes: int = 3) -> None:
        super().__init__()
        
        self.cnn = CNNFeatureExtractor(in_channels, cnn_channels)
        self.gru = GRUEncoder(cnn_channels, gru_hidden)
        self.attention = TemporalAttention(gru_hidden)
        self.head = ClassificationHead(gru_hidden, num_classes)

    def forward(self, x: torch.Tensor, return_scores: bool = False) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        
        cnn_out = self.cnn(x)
        gru_out = self.gru(cnn_out)
        context = self.attention(gru_out)
        logits = self.head(context)

        return (logits, F.softmax(logits, dim=-1)) if return_scores else logits