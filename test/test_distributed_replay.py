import unittest
import tensorflow as tf
import numpy as np
import multiprocessing

import time
from dtf.cluster import Cluster
from dtf.replay_buffer import ReplayBuffer, DistributedRelayBuffer

from test_utils import wait_for_shutdown

physical_devices = tf.config.list_physical_devices('GPU')
try:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass


def distributed_fn(args):
    cluster_dict, task, task_idx = args

    clus = Cluster(cluster_dict, task, task_idx)
    clus.start()

    model = DistributedRelayBuffer(
        "replay_buffer", {"test": (1,1)},
        2, 2, "worker", "learner", "replay",
        task, task_idx, clus)

    if task == "replay":
        try:
            model.run()
        except:
            pass

    elif task == "worker":
        model.push({"test": np.array([[1.]])})
        for k in model._update_queues:
            print(k, model._update_queues[k].size())
    elif task == "learner":
        data = model.pull(return_data=True)
        print(data)
    wait_for_shutdown(model)

class TestDistributedReplayBuffer(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
    def test_distributed_replay(self):
        cluster_dict = {'learner': ['localhost:6006'],
                        'worker': ['localhost:6007'],
                        'replay': ['localhost:6008']}
        with multiprocessing.Pool(3) as pool:
            # cluster_dict, module_name, task, task_idx,
            # source, sink, push_to_all
            pool.map(distributed_fn, [(cluster_dict, 'learner', 0),
                                    (cluster_dict, 'worker', 0),
                                    (cluster_dict, 'replay', 0)])
if __name__ == "__main__":
    unittest.main()
