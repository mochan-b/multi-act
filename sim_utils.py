import numpy as np

class NumpyRingBuffer:
    """
    A fixed‑capacity ring buffer of homogeneous numpy arrays.
    - capacity: max number of elements
    - shape: shape of each element (e.g. (14,) or (3,480,640))
    - dtype: array dtype
    """
    
    def __init__(self, capacity, shape, dtype=np.float32):
        """
        A fixed‑capacity ring buffer of homogeneous numpy arrays.
        - capacity: max number of elements
        - shape: shape of each element (e.g. (14,) or (3,480,640))
        - dtype: array dtype
        """
        self.data = np.empty((capacity, *shape), dtype=dtype)
        self.capacity = capacity
        self.head = 0      # next write position
        self.size = 0      # how many elements have been appended (≤ capacity)

    def append(self, x):
        """Append one element x of shape `shape`. Overwrites oldest when full."""
        self.data[self.head] = x
        self.head = (self.head + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get(self, indices):
        """
        Fetch elements by "reverse‑relative" index:
          idx = 0 → most recently appended,
          idx = 1 → one before that, etc.
        - indices: array‑like of ints
        - if any idx >= capacity → IndexError
        - if idx >= size (but < capacity) → returns the oldest element for that idx
        """
        if self.size == 0:
            raise ValueError("Buffer is empty; no elements to get.")
        
        # turn into numpy array of integers
        idx = np.asarray(indices, dtype=int)
        
        # check for invalid values
        if np.any(idx < 0):
            raise IndexError("Negative indices not supported.")
        if np.any(idx >= self.capacity):
            raise IndexError(f"indices must be < capacity ({self.capacity})")
        
        # clamp any idx beyond current size to the oldest element (size-1)
        clamp = np.minimum(idx, self.size - 1)
        
        # compute buffer positions: head-1 is last written
        # so position = (head - 1 - clamp) mod capacity
        pos = (self.head - 1 - clamp) % self.capacity
        
        # fancy‑index into self.data
        return self.data[pos]
