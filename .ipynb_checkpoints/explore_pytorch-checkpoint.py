import torch
import numpy as np

data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")
print(x_ones.dtype)

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

x_ones1 = x_ones.type(torch.float)
print(f"ones float {x_ones1}")
print(x_ones1.dtype)

# convert tensor into numpy
y = x_ones.numpy()
print("convert tensor into numpy array ")
print(y)
print(y.dtype)

# convert numpy array into tensor
x = np.zeros((2,2),dtype=np.float32)
print(x)
print(x.dtype)
print("convert numpy array into tensor")

y=torch.from_numpy(x)
print(y)
print(y.dtype)

# Moving tensor between device
print("Moving tensor between device")
x = torch.tensor([1.5,2])
print(x)
print(x.device)
#  change tenor to cuda
if torch.cuda.is_available():
    device = torch.device("cuda:0")
x = x.to(device)
print(x)
print(x.device)
# change tensor to cpu
device = torch.device("cpu")
x = x.to(device)
print(x)
print(x.device)
# create tensor on any device directly
device = torch.device("cuda:0")
x = torch.ones(2,2,device=device)
print(x)