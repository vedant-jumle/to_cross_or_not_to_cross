from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torchvision.models as tv_models


# ---------------------------------------------------------------------------
# ResNet18 backbone
# ---------------------------------------------------------------------------

class ResNet18Backbone(nn.Module):
    """
    Pretrained ResNet18 as a per-frame feature extractor.
    Removes the final FC layer; outputs (N, 512) after global average pool.

    Args:
        pretrained:  Load ImageNet weights (default True).
        frozen:      Freeze all parameters at init (default True).
                     Call unfreeze() to enable fine-tuning.
    """

    def __init__(self, pretrained: bool = True, frozen: bool = True) -> None:
        super().__init__()
        base = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        # Keep everything except the final FC
        self.features = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = 512

        if frozen:
            for p in self.parameters():
                p.requires_grad = False

    def unfreeze(self, from_layer: int = 0) -> None:
        """
        Re-enables gradients for backbone layers from `from_layer` onward.
        Layers are indexed 0-7 (conv1, bn1, relu, maxpool, layer1-4).
        Common use: from_layer=4 to fine-tune layer1–4, from_layer=6 for layer3–4 only.
        """
        for i, module in enumerate(self.features):
            if i >= from_layer:
                for p in module.parameters():
                    p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, 3, H, W) float32 in [0, 1]
        Returns:
            (N, 512)
        """
        x = self.features(x)
        x = self.pool(x)
        return x.flatten(1)


# ---------------------------------------------------------------------------
# Temporal Transformer
# ---------------------------------------------------------------------------

class TemporalTransformer(nn.Module):
    """
    Transformer encoder over T frame features.
    Input dim matches ResNet18 output (512) so no projection is needed.

    Args:
        d_model:        Feature dimension (default 512, matches ResNet18).
        nhead:          Attention heads (default 4).
        num_layers:     Encoder depth (default 2).
        dim_feedforward: FFN hidden size (default 1024).
        dropout:        Dropout rate (default 0.1).
        T:              Temporal window; used to initialise positional embedding.
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        T: int = 8,
    ) -> None:
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, T, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # pre-LN: more stable training
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_dim = d_model

    def forward(self, frame_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frame_feats: (B, T, 512)
        Returns:
            (B, 512) — mean-pooled temporal representation
        """
        x = frame_feats + self.pos_embed
        x = self.encoder(x)
        return x.mean(dim=1)


# ---------------------------------------------------------------------------
# Tabular MLP
# ---------------------------------------------------------------------------

class TabularMLP(nn.Module):
    """
    Small MLP for tabular features.

    Args:
        in_dim:  Input feature count (default 9).
        hidden:  Hidden layer size (default 64).
        out_dim: Output size (default 64).
        dropout: Dropout rate (default 0.1).
    """

    def __init__(
        self,
        in_dim: int = 9,
        hidden: int = 64,
        out_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.ReLU(),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_dim)
        Returns:
            (B, out_dim)
        """
        return self.net(x)


# ---------------------------------------------------------------------------
# Full multi-modal model
# ---------------------------------------------------------------------------

class VisualCrossingPredictor(nn.Module):
    """
    Multi-modal pedestrian crossing predictor.

    Streams:
      - Visual:  ResNet18 per frame → TemporalTransformer → (B, 512)
      - Tabular: TabularMLP → (B, 64)
    Fusion: concat → MLP → raw logit (B,)

    Output is a raw logit — apply torch.sigmoid externally for probabilities,
    or use BCEWithLogitsLoss directly during training.

    Args:
        T:               Temporal window (must match VisualPIEDataset.T).
        n_tab_features:  Number of tabular input features (default 9).
        backbone_frozen: Freeze ResNet18 at init (default True).
        d_model:         Transformer internal dimension (default 512).
        nhead:           Transformer attention heads (default 4).
        num_layers:      Transformer encoder depth (default 2).
        tab_hidden:      Tabular MLP hidden size (default 64).
        fusion_hidden:   Fusion MLP hidden size (default 128).
        dropout:         Dropout in fusion head (default 0.2).
    """

    def __init__(
        self,
        T: int = 8,
        n_tab_features: int = 9,
        backbone_frozen: bool = True,
        d_model: int = 512,
        nhead: int = 4,
        num_layers: int = 2,
        tab_hidden: int = 64,
        fusion_hidden: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.T = T

        self.backbone = ResNet18Backbone(pretrained=True, frozen=backbone_frozen)
        self.temporal = TemporalTransformer(
            d_model=d_model, nhead=nhead, num_layers=num_layers, T=T,
        )
        self.tab_mlp = TabularMLP(in_dim=n_tab_features, hidden=tab_hidden, out_dim=tab_hidden)

        fusion_in = self.temporal.out_dim + self.tab_mlp.out_dim  # 512 + 64 = 576
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1),
        )

    def forward(self, x_tab: torch.Tensor, x_vis: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_tab: (B, 9)              tabular features, float32
            x_vis: (B, T, 3, H, W)    cropped frame sequence, float32 in [0, 1]
        Returns:
            logits: (B,)  raw logits
        """
        B, T, C, H, W = x_vis.shape

        # Visual stream
        frame_feats = self.backbone(x_vis.view(B * T, C, H, W))  # (B*T, 512)
        frame_feats = frame_feats.view(B, T, -1)                  # (B, T, 512)
        visual_out = self.temporal(frame_feats)                    # (B, 512)

        # Tabular stream
        tab_out = self.tab_mlp(x_tab)                             # (B, 64)

        # Fusion
        fused = torch.cat([visual_out, tab_out], dim=1)           # (B, 576)
        logits = self.fusion(fused).squeeze(-1)                   # (B,)
        return logits

    def unfreeze_backbone(self, from_layer: int = 0) -> None:
        """Unfreeze ResNet18 layers starting from `from_layer`."""
        self.backbone.unfreeze(from_layer)

    def parameter_groups(
        self,
        backbone_lr: float = 1e-5,
        head_lr: float = 1e-3,
    ) -> List[dict]:
        """
        Returns optimizer parameter groups with separate learning rates:
          - backbone parameters: lr=backbone_lr  (use when unfreezing)
          - temporal + tabular + fusion: lr=head_lr

        Usage:
            optimizer = torch.optim.AdamW(model.parameter_groups())
        """
        backbone_params = list(self.backbone.parameters())
        backbone_ids = {id(p) for p in backbone_params}
        head_params = [p for p in self.parameters() if id(p) not in backbone_ids]
        return [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ]
