import numpy as np 

class Neuron:
    
    def setBias(self , b):
        """Set Bias
                Args:
                    b: bias
        """
        self.bias = b
    
    def setWeight(self, w):
        """Set Weight
                Args:
                    w: weight
        """
        self.weights = w
    
    def __init__(self, weights , bias):
        """Constructor Function
                Args:
                    weights: Weights for model.
                    bias: bias for model.
        """
        self.weights = weights
        self.bias    = bias
        
    def activationSigmoid(self, x):
        """Activation Function
                Args:
                    x: for Sigmoid
        """
        return 1 / (1+np.exp(-x))
    
    def inverseSigmoid(self, x):
        sig = self.activationSigmoid(x)
        return sig*(1-sig)
    
    def feedSum(self, datas):
        """Sum Data
                Args:
                    datas:
        """
        totalCalc = 0
        for data , weight in zip(datas,self.weights):
            totalCalc += data*weight
        
        return (totalCalc+self.bias)
    
    def forward(self, datas):
        """Forward
                Args:
                    datas:
        """
        total = self.feedSum(datas)
        return self.activationSigmoid(total)

    def __repr__(self):
        return f"Neuron(weights={self.weights}, bias={self.bias})"
    
n = Neuron( [20] , 5 )
print("Neuron: " , n)

y = n.forward([1])
print("Forward 1: " , y)

c = n.forward([0.2])
print("Forward 2: " , c)