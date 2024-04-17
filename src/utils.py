from typing import Union
import numpy as np

def calculate_cnn_output_dim(input_size, kernel_sizes, strides, paddings):
    output_size = input_size

    for kernel_size, stride, padding in zip(kernel_sizes, strides, paddings):
        output_size = ((output_size - kernel_size + 2 * padding) // stride) + 1

    return output_size

class EnhancedOrderedDict():
    def __init__(self, max_size: int):
        self.set = set()
        self.keys = []
        self.size = 0 # used for lazy deletion of keys
        self.tail = 0
        self.max_size = max_size

    # use head and tail for array to handle lazy deletion for O(1) time complexity
    def insert(self, key: Union[str, int]):
        wrapped = False
        if len(self.keys) == self.max_size:
            # delete the tail element
            if self.size == self.max_size:
                self.set.remove(self.keys[self.tail])
                self.tail = (self.tail + 1) % self.max_size

            self.set.add(key)
            print(self.tail, self.size)
            head = (self.tail + self.size - (self.size == self.max_size)) % self.max_size
            self.keys[head] = key
            self.size = min(self.size + 1, self.max_size)

        else:
            self.set.add(key)
            self.keys.append(key)
            self.size += 1
        
    def pop(self):
        """
        Remove from tail
        """
        if self.size == 0:
            return None
        
        key = self.keys[self.tail]
        self.set.remove(key)
        self.tail = (self.tail + 1) % self.max_size
        self.size -= 1
        return key
    
    def has(self, key: Union[str, int]) -> bool:
        return key in self.set
        
    def sample(self, n_samples: int = 1):
        cut = np.random.choice(self.size, n_samples, replace=False)
        idx_in_keys = (cut + self.tail) % self.max_size
        return [self.keys[i] for i in idx_in_keys]