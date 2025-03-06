import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from core.models_dual_network import DualBranchNetwork
from main.main_dual_network import compute_pde_loss

def test_gradients():
    # 創建一個小的測試數據集
    batch_size = 2
    seq_len = 10
    x_convlstm = torch.randn(batch_size, 3, seq_len, 5, requires_grad=True)
    x_transformer = torch.randn(batch_size, seq_len, 2, requires_grad=True)
    y= torch.rand(batch_size, requires_grad=True)
    
    # 初始化模型
    model = DualBranchNetwork()
    
    # 確保模型處於訓練模式
    model.train()
    
    # 前向傳播
    outputs = model(x_convlstm, x_transformer)
    V, t, S = outputs
    
    
    # 確保 V 需要梯度
    # if not V.requires_grad:
    #     V = V.detach().requires_grad_()
    
    # 測試梯度計算
    sigma = x_convlstm[:, 0, -1, 3]
    r = 0.0079
    
    
    # 計算 PDE loss
    pde_loss = compute_pde_loss(V,t,S,sigma,r)
    
    # 確保 t 和 S 保留梯度
    # t.retain_grad()
    # S.retain_grad()
    
    # 驗證梯度是否存在且可以反向傳播
    pde_loss.backward()
    
    # 檢查梯度是否成功計算
    print("Loss value:", pde_loss.item())
    print("\nt tensor:", t)
    print("t requires_grad:", t.requires_grad)
    print("t grad:", t.grad)
    
    print("\nS tensor:", S)
    print("S requires_grad:", S.requires_grad)
    print("S grad:", S.grad)
        
    return True
        

if __name__ == "__main__":
    test_gradients()
