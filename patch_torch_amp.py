"""
Patch torch.amp to be compatible with fastai 2.x and PyTorch 2.2.2
This addresses the import incompatibility where fastai expects torch.amp.GradScaler
but PyTorch 2.2.2 has it in torch.cuda.amp.GradScaler
"""
import sys
import torch
import torch.cuda.amp as cuda_amp

# Patch torch.amp module to include GradScaler and related classes
torch.amp.GradScaler = cuda_amp.GradScaler
torch.amp.grad_scaler = cuda_amp.grad_scaler

print("✓ Applied torch.amp compatibility patch for fastai+PyTorch 2.2.2")
