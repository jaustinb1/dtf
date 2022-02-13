import tensorflow as tf
import numpy as np

from dtf.modules import DistributedModel


class ReplayBuffer(DistributedModel):

    def __init__(self, storage_spec, capacity):
        super().__init__(update_method='scatter')
        self.storage_spec = storage_spec
        self.capacity = capacity

        self.size = -1

        self.storage = {}
        for k, shape in self.storage_spec.items():
            self.storage[k] = tf.Variable(
                shape=(self.capacity, *shape),
                name=k,
                initial_value=np.zeros([self.capacity, *shape]),
                dtype=tf.float32)

    def clear(self):
        self.size = min(0, self.size)

    def get_scatter_index(self):
        if self.size < self.capacity:
            self.size = max(self.size, 0)
            idx = self.size
            self.size += 1
        else:
            idx = np.random.randint(0, self.size)

        return idx

    def sample(self):
        return self.sample_n(1)

    def sample_n(self, n):
        if self.size == -1:
            return None
        if self.size < n:
            print("WARNING: sampling a larger batch than we have samples")
        idx = np.random.randint(0, self.size, size=n)
        ret = {}
        for k in self.storage:
            ret[k] = tf.gather(self.storage[k], idx)
        return ret

    def add(self, to_add):
        batch_dim = -1
        for k in to_add:
            dim = to_add[k].shape[0]
            if batch_dim == -1:
                batch_dim = dim
            else:
                assert dim == batch_dim
        assert batch_dim > 0

        idx = [[self.get_scatter_index()] for _ in range(batch_dim)]

        for k in self.storage:
            assert k in to_add, f"You are missing {k} in your data"
            self.storage[k] = tf.tensor_scatter_nd_update(
                self.storage[k], idx, to_add[k])
