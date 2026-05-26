"""
LOB Price Movement Prediction Model
====================================
Research-grade baseline inspired by DeepLOB for 3-class classification
(DOWN=0, FLAT=1, UP=2) on Limit Order Book event sequences.

Input shape:  (batch, 100, 40)  — 100 time steps, 40 LOB features
Output shape: (batch, 3)        — raw logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. CNN Feature Extractor
# ---------------------------------------------------------------------------

class CNNFeatureExtractor(nn.Module):
    """
    Extracts local microstructure patterns (spread, imbalance, liquidity)
    from the raw LOB feature sequence using 1-D convolutions.

    Input:  (batch, seq_len, in_channels=40)
    Output: (batch, seq_len, 64)

    Includes a residual projection so the skip connection always matches
    the output channel dimension.
    """

    def __init__(self, in_channels: int = 40, hidden_channels: int = 64) -> None:
        super().__init__()

        # --- Block ---
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(hidden_channels)

        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(hidden_channels)

        # Residual projection: maps in_channels → hidden_channels (1×1 conv)
        self.residual_proj = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, 40)  →  permute for Conv1d: (batch, 40, seq_len)
        x = x.permute(0, 2, 1)                      # (B, 40, T)

        residual = self.residual_proj(x)             # (B, 64, T)

        out = F.gelu(self.bn1(self.conv1(x)))        # (B, 64, T)
        out = F.gelu(self.bn2(self.conv2(out)))      # (B, 64, T)

        out = out + residual                          # residual connection

        out = out.permute(0, 2, 1)                   # (B, T, 64)  — back to batch-first
        return out


# ---------------------------------------------------------------------------
# 2. GRU Encoder
# ---------------------------------------------------------------------------

class GRUEncoder(nn.Module):
    """
    Models temporal market dynamics over the CNN feature sequence.

    Input:  (batch, seq_len, 64)
    Output: (batch, seq_len, 128)   — full sequence of hidden states

    LayerNorm is applied after the GRU for training stability.
    """

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
        # x: (B, T, 64)
        out, _ = self.gru(x)          # out: (B, T, 128)
        out = self.layer_norm(out)    # (B, T, 128)
        return out


# ---------------------------------------------------------------------------
# 3. Temporal Attention
# ---------------------------------------------------------------------------

class TemporalAttention(nn.Module):
    """
    Learnable soft-attention over the time dimension.
    The model learns to weight which time steps are most informative
    for the prediction, yielding a single context vector per sample.

    Input:  (batch, seq_len, hidden_size=128)
    Output: (batch, hidden_size=128)
    """

    def __init__(self, hidden_size: int = 128) -> None:
        super().__init__()

        # Learnable projection: hidden_size → 1 score per time step
        self.attention_proj = nn.Linear(hidden_size, 1, bias=True)

        nn.init.xavier_uniform_(self.attention_proj.weight)
        nn.init.zeros_(self.attention_proj.bias)

    def forward(self, gru_out: torch.Tensor) -> torch.Tensor:
        # gru_out: (B, T, 128)

        scores = self.attention_proj(gru_out)           # (B, T, 1)
        weights = F.softmax(scores, dim=1)              # (B, T, 1)  — softmax over time

        # Weighted sum of GRU hidden states
        context = (weights * gru_out).sum(dim=1)        # (B, 128)
        return context


# ---------------------------------------------------------------------------
# 4. Classification Head
# ---------------------------------------------------------------------------

class ClassificationHead(nn.Module):
    """
    Maps the attention context vector to 3-class logits.

    Input:  (batch, 128)
    Output: (batch, 3)  — raw logits (use CrossEntropyLoss directly)
    """

    def __init__(self, hidden_size: int = 128, num_classes: int = 3) -> None:
        super().__init__()

        self.fc1     = nn.Linear(hidden_size, 64)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.fc2     = nn.Linear(64, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in [self.fc1, self.fc2]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 128)
        out = self.relu(self.fc1(x))   # (B, 64)
        out = self.dropout(out)        # (B, 64)
        out = self.fc2(out)            # (B, 3)
        return out


# ---------------------------------------------------------------------------
# 5. Full LOB Model
# ---------------------------------------------------------------------------

class LOBModel(nn.Module):
    """
    End-to-end LOB price movement predictor.

    Pipeline:
        Input (B, 100, 40)
          → CNNFeatureExtractor  → (B, 100, 64)
          → GRUEncoder           → (B, 100, 128)
          → TemporalAttention    → (B, 128)
          → ClassificationHead  → (B, 3)

    Args:
        in_channels   : number of LOB features per time step (default 40)
        cnn_channels  : CNN hidden channel width              (default 64)
        gru_hidden    : GRU hidden state size                 (default 128)
        num_classes   : number of output classes              (default 3)

    Returns:
        logits (B, 3) — raw, unnormalised class scores.
        Pass through nn.CrossEntropyLoss or F.softmax as needed.
    """

    def __init__(
        self,
        in_channels: int = 40,
        cnn_channels: int = 64,
        gru_hidden: int = 128,
        num_classes: int = 3,
    ) -> None:
        super().__init__()

        self.cnn       = CNNFeatureExtractor(in_channels, cnn_channels)
        self.gru       = GRUEncoder(cnn_channels, gru_hidden)
        self.attention = TemporalAttention(gru_hidden)
        self.head      = ClassificationHead(gru_hidden, num_classes)

    def forward(self, x: torch.Tensor, return_scores: bool = False) -> torch.Tensor:
        """
        Args:
            x             : (B, T, 40)  LOB sequence tensor
            return_scores : if True, also return softmax probabilities

        Returns:
            logits  : (B, 3)
            scores  : (B, 3)  [only when return_scores=True]
        """
        # (B, T, 40) → (B, T, 64)
        cnn_out = self.cnn(x)

        # (B, T, 64) → (B, T, 128)
        gru_out = self.gru(cnn_out)

        # (B, T, 128) → (B, 128)
        context = self.attention(gru_out)

        # (B, 128) → (B, 3)
        logits = self.head(context)

        if return_scores:
            return logits, F.softmax(logits, dim=-1)
        return logits


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size, seq_len, lob_features = 32, 100, 40
    dummy_input = torch.randn(batch_size, seq_len, lob_features)

    model = LOBModel()
    model.eval()

    with torch.no_grad():
        logits, scores = model(dummy_input, return_scores=True)

    print(f"Input shape  : {dummy_input.shape}")
    print(f"Logits shape : {logits.shape}")
    print(f"Scores shape : {scores.shape}")
    print(f"Score sum    : {scores.sum(dim=-1).mean().item():.6f}  (should be 1.0)")

    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params : {total_params:,}")
    print(f"Trainable    : {trainable:,}")
