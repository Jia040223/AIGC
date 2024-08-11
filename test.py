import torch

A = torch.tensor([[1, 2], [3, 4]])
B = torch.tensor([[5, 6]])
result = torch.einsum('ij,ik->ijk', A, B)
print(result)  # 输出: tensor([[[ 5,  6], [10, 12]], [[15, 18], [20, 24]]])