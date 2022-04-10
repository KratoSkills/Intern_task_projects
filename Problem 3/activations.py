import jax.numpy as jnp

def sigmoid(Z):
    A = 1/(1 + jnp.exp(-Z))
    return A

def relu(Z):
    A = jnp.maximum(0,Z)
    return A

def softmax(Z):
    exps = jnp.exp(Z - jnp.max(Z))  
    for i in range(len(Z)):
      exps /= (exps.sum(axis=0,keepdims=True))  

    return exps

def sigmoid_derivative(Z):
    s = sigmoid(Z)  
    return s * (1 - s)

def relu_backward(dA,Z):
    dZ = jnp.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ = dZ.at[Z <= 0].set(0)
    # dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ