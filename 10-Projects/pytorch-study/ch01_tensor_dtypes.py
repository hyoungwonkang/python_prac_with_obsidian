import torch

tensor0d = torch.tensor(1)

tensor1d = torch.tensor([1, 2, 3])
floatvec = tensor1d.to(torch.float32)
print(tensor1d.dtype)
print(floatvec.dtype)

tensor2d = torch.tensor([[1, 2], 
                         [3, 4]])

tensor3d = torch.tensor([[[1, 2], [3, 4]], 
                         [[5, 6], [7, 8]]])
