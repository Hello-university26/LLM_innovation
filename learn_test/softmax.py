
import torch
import torch.nn.functional as F

# 模拟三个类别的原始分数
logits = torch.tensor([2.5, 0.8, 1.2])

# 使用Softmax转换为概率
probabilities = F.softmax(logits, dim=0)

print("原始分数：", logits)

print("概率：", probabilities)

print("概率总和：", probabilities.sum())
