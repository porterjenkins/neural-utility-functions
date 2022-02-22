import torch


x = torch.ones(2, 2, requires_grad=True)
print("x", x)
y = x + 2
print("y", y)

z = y * y * 3
print("z", z)

out1 =  z.sum()
print("out1", out1)

out1.backward(retain_graph=True)
print("x grad", x.grad)


out2 = z.mean()
print("out2", out2)

out2.backward(retain_graph=True)
print("x grad", x.grad)

out3 = z.sum()

dz_dx = torch.autograd.grad(out3, x, create_graph=True)[0]
print(dz_dx)