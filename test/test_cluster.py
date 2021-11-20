import unittest
import multiprocessing
from dtf.cluster import Cluster
import tensorflow as tf

def start_cluster(args):
    cluster_dict, name, task, assert_devices = args
    clus = Cluster(cluster_dict, name, task)
    clus.start()
    if assert_devices:
        devices = [d.name for d in tf.config.list_logical_devices()]
        for task_name in cluster_dict:
            for i in range(len(cluster_dict[task_name])):
                assert f"/job:{task_name}/replica:0/task:{i}/device:CPU:0" in devices
                assert f"/job:{task_name}/replica:0/task:{i}/device:GPU:0" in devices


class TestCluster(unittest.TestCase):

    def test_cluster_start(self):
        cluster_dict = {'learner': ['localhost:6006'],
                        'worker': ['localhost:6007']}
        with multiprocessing.Pool(2) as pool:
            pool.map(start_cluster, [(cluster_dict, 'learner', 0, False),
                                     (cluster_dict, 'worker', 0, False)])
        print("Done")

    def test_cluster_devices(self):
        cluster_dict = {'learner': ['localhost:6006'],
                        'worker': ['localhost:6007']}
        with multiprocessing.Pool(2) as pool:
            pool.map(start_cluster, [(cluster_dict, 'learner', 0, True),
                                     (cluster_dict, 'worker', 0, True)])

        print("Done")

if __name__ == "__main__":
    unittest.main()
