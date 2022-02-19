import unittest
import tensorflow as tf
import numpy as np
import multiprocessing

import time
from dtf.replay_buffer import ReplayBuffer

physical_devices = tf.config.list_physical_devices('GPU')
try:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

class TestReplayBuffer(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)

    def test_base_replay(self):
        replay_buffer = ReplayBuffer({'test': [1, 1]}, 2)
        replay_buffer.clear()
        assert replay_buffer.size == -1
        sample = replay_buffer.sample()
        assert sample is None
        replay_buffer.add({'test': tf.constant([[1.]])})

        # Test insertion
        assert np.all(np.equal(
            replay_buffer.storage['test'].numpy(),
            np.array([[[1.]], [[0.]]])
        ))

        # Test sampling single element from size=1 buffer
        sample = replay_buffer.sample()
        assert np.all(np.equal(
            sample['test'].numpy(),
            np.array([[[1.]]])
        ))

        # Test filling the replay buffer
        replay_buffer.add({'test': tf.constant([[3.]])})
        assert np.all(np.equal(
            replay_buffer.storage['test'].numpy(),
            np.array([[[1.]], [[3.]]])
        ))

        # Test evicting
        replay_buffer.add({'test': tf.constant([[2.]])})
        assert np.all(np.equal(
            replay_buffer.storage['test'].numpy(),
            np.array([[[2.]], [[3.]]])
        ))
        print("done")

if __name__ == "__main__":
    unittest.main()
