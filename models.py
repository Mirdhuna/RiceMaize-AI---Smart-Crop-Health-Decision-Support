from pathlib import Path

import torch
from torch import nn


DEIT_TINY_URL = (
    "https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth"
)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class RiceMaizeCNN(nn.Module):
    """Baseline CNN for leaf spots, lesions, discoloration, and pest texture."""

    def __init__(self, num_classes=17):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=3, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        batch, tokens, dim = x.shape
        qkv = (
            self.qkv(x)
            .reshape(batch, tokens, 3, self.num_heads, dim // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(batch, tokens, dim)
        return self.proj_drop(self.proj(x))


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class VisionTransformer(nn.Module):
    """DeiT-Tiny compatible ViT: 16x16 patches, attention over local and global parts."""

    def __init__(
        self,
        num_classes=17,
        img_size=224,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4.0,
        drop_rate=0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim)
        )
        self.pos_drop = nn.Dropout(drop_rate)
        self.blocks = nn.Sequential(
            *[
                Block(embed_dim, num_heads, mlp_ratio=mlp_ratio, drop=drop_rate)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_module)

    @staticmethod
    def _init_module(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.norm(self.blocks(x))
        return self.head(x[:, 0])


def load_deit_tiny_weights(model, cache_dir="checkpoints"):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = cache_dir / "deit_tiny_patch16_224.pth"

    if not checkpoint_path.exists():
        torch.hub.download_url_to_file(DEIT_TINY_URL, str(checkpoint_path), progress=True)

    payload = torch.load(checkpoint_path, map_location="cpu")
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    filtered = {
        key: value
        for key, value in state.items()
        if not key.startswith("head")
    }
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    return missing, unexpected


def freeze_vit_backbone(model, train_last_blocks=4):
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.head.parameters():
        parameter.requires_grad = True

    for parameter in model.norm.parameters():
        parameter.requires_grad = True

    start = max(0, len(model.blocks) - train_last_blocks)
    for block in model.blocks[start:]:
        for parameter in block.parameters():
            parameter.requires_grad = True


def build_model(name, num_classes=17, pretrained=True):
    name = name.lower()
    if name == "cnn":
        return RiceMaizeCNN(num_classes=num_classes)

    if name == "vit":
        model = VisionTransformer(num_classes=num_classes)
        if pretrained:
            load_deit_tiny_weights(model)
            freeze_vit_backbone(model, train_last_blocks=4)
        return model

    raise ValueError("name must be 'cnn' or 'vit'")


class HybridEnsemble(nn.Module):
    """Validation-tuned mix of CNN and ViT class probabilities."""

    def __init__(self, cnn, vit, cnn_weight=0.5):
        super().__init__()
        self.cnn = cnn
        self.vit = vit
        self.cnn_weight = float(cnn_weight)

    def forward(self, x):
        cnn_logits = self.cnn(x)
        vit_logits = self.vit(x)
        cnn_prob = torch.softmax(cnn_logits, dim=1)
        vit_prob = torch.softmax(vit_logits, dim=1)
        mixed = self.cnn_weight * cnn_prob + (1.0 - self.cnn_weight) * vit_prob
        return mixed

    def predict(self, x):
        mixed = self.forward(x)
        confidence, predicted = mixed.max(dim=1)
        return predicted, confidence, mixed
