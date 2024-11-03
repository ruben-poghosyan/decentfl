import numpy as np
def x2(x):
    return x**3
def minmax_scale(data, feature_range=(0, 1)):
    """
    Scales the input data using MinMax scaling.

    Parameters:
    - data: array-like, the data to scale.
    - feature_range: tuple (min, max), default is (0, 1) for scaling.

    Returns:
    - Scaled data.
    """
    data = np.array(data)
    min_val = np.min(data)
    max_val = np.max(data)
    
    # Scale the data
    scaled_data = feature_range[0] + (data - min_val) * (feature_range[1] - feature_range[0]) / (max_val - min_val)
    
    return scaled_data
x = np.linspace(-10,10,20)
y = x2(x)
print(x,y, minmax_scale(y))