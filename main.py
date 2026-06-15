import mlx.core as mx

def normalize_and_grad(x: mx.array):
    """
    1. Compute mean and std of x
    2. Normalize: x_norm = (x - mean) / std
    3. Define loss = sum(x_norm ** 2)
    4. Compute gradient of loss w.r.t. x
    5. Return (x_norm, gradient)
    """
    mean = x.mean()
    std = x.std()
    x_norm = (x - mean) / (std + 1e-8)
    
    def norm_then_loss(x_in: mx.array):
      m = x_in.mean()
      s = x_in.std()
      n = (x_in - m) / (s + 1e-8)
      return (n ** 2).sum()

    dL_dx = mx.grad(norm_then_loss)(x)
    return x_norm, dL_dx

x = mx.array([2.0, 4.0, 6.0, 8.0])
x_norm, grad = normalize_and_grad(x)
print(x_norm)  # should be roughly [-1.34, -0.45, 0.45, 1.34]
print(grad)    # what do you expect here — and why?