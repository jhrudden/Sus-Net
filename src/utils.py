from typing import Union, Any
import torch

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
        if self.size + 1 > self.max_size:
            # delete the tail element
            self.set.remove(self.keys[self.tail])
            self.tail = (self.tail + 1) % self.max_size
            self.size -= 1
        
        # NOTE: going to assume that the key is unique
        self.set.add(key)
        # append at tail if size is less than max_size
        if self.tail > 0:
            self.keys[self.tail-1] = key
            self.size += 1
        else:
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
        cut = torch.randint(0, self.size, (n_samples,))
        idx_in_keys = (cut + self.tail) % self.max_size
        return [self.keys[i] for i in idx_in_keys]