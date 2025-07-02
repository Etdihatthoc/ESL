from train import ESLGradingModel
import torch
import torch.nn.functional as F

model = ESLGradingModel.load('./model/model.pth')

if hasattr(model, 'attn_pool') and hasattr(model.attn_pool, 'scale'):
    print(f"attn_pool scale = {model.attn_pool.scale.item()}")