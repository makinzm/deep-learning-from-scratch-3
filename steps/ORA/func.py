import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        self.input = input
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx

A = Square()

x = Variable(np.array(2.0))
a = A(x)
print(A.backward(500))

B=Square()
B.input=Variable(np.array(2.0))
print(B.backward(1))

#print(Square().backward(1))
#AttributeError
