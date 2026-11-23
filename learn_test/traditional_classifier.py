
import torch
import torch.nn as nn

# 创建一个传统分类头
classifier = nn.Linear(
    in_features=384,
    out_features=3
)

# 模拟 Encoder 输出的384维向量
x = torch.randn(1, 384)

# 送入分类头
logits = classifier(x)

print("分类原始分数：")
print(logits)

print("输出形状：")
print(logits.shape)
