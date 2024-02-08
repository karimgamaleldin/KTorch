from autograd.Tensor import Tensor
import torch 

a = Tensor([1.0, 2.0, 3.0], label='a')
b = Tensor([1.0, 2.0, -3.0], label='b')
c = Tensor([4.0, 5.0, 6.0], label='c')
e = a * b; e.label = 'e' 
d = e + c; d.label = 'd'
L = d ** 2; L.label = 'L'

print(a)
print(b)
print(c)
print(d)
print(e)

print(L)
L.backward()
print(L.grad)
print(d.grad)
print(e.grad)
print(a.grad)
print(b.grad)
print(c.grad)

a = Tensor([[1, 2], [3, 4]], label='a')
b = Tensor([[5, 6], [7, 8]], label='b')
c = a.matmul(b)

c.backward()
print(a.grad)
print(b.grad)
print(c.grad)


a = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
b = torch.tensor([[5., 6.], [7., 8.]], requires_grad=True)
c = a @ b

c.backward(gradient=torch.tensor([[1., 1.], [1., 1.]]))
print(a.grad)
print(b.grad)
