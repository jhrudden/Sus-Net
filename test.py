import numpy as np
import math

# Example array of shape (n, m, b)
n, m, b = 2, 3, 4  # Example dimensions
arr = np.arange(n * m * b).reshape(n, m, b)  # Example array

# Target size
x = 5  # Example target size

# Calculate repeat factors
repeat_factor_m = math.ceil(x / m)
repeat_factor_b = math.ceil(x / b)

# Repeat and slice to match the target size exactly
blown_up_arr_m = np.repeat(arr, repeat_factor_m, axis=1)[
    :, :x, :
]  # Blow up and slice m
blown_up_arr_mb = np.repeat(blown_up_arr_m, repeat_factor_b, axis=2)[
    :, :, :x
]  # Blow up and slice b

print("Original shape:", arr.shape)
print("New shape:", blown_up_arr_mb.shape)
