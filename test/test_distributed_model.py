import unittest
import tensorflow as tf
import multiprocessing
import time
from dtf.cluster import Cluster
from dtf.modules import DistributedModule, DistributedModel
import numpy as np

from test_utils import wait_for_shutdown

physical_devices = tf.config.list_physical_devices('GPU')
try:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass


class DummyModel(DistributedModel):
    def __init__(self):
        super().__init__()
        self.v = tf.Variable(0.0)

    def __call__(self, x):
        return x * self.v

def testpush_fn(args):
    cluster_dict, module_name, task, task_idx, \
        source, sink, push_to_all = args

    clus = Cluster(cluster_dict, task, task_idx)
    clus.start()

    model = DistributedModule(DummyModel(),
                              cluster=clus,
                              data_source=source,
                              module_name=module_name,
                              task=task,
                              index=task_idx,
                              push_to_all=push_to_all)

    if task == source:
        model._base_model.v.assign(1.0)
        if push_to_all:
            model.push()
        else:
            model.push(force_ind=0)
            model.push(force_ind=1)
        for k in model._update_queues:
            assert model._update_queues[k].size() == 1
            model._update_queues[k].dequeue()
        wait_for_shutdown(model)
    else:
        time.sleep(3)
        wait_for_shutdown(model)

def testpull_fn(args):
    cluster_dict, module_name, task, task_idx, \
        source, sink, push_to_all = args

    clus = Cluster(cluster_dict, task, task_idx)
    clus.start()

    model = DistributedModule(DummyModel(),
                              cluster=clus,
                              data_source=source,
                              module_name=module_name,
                              task=task,
                              index=task_idx,
                              push_to_all=push_to_all)

    if task == source:
        model._base_model.v.assign(1.0)
        if push_to_all:
            model.push()
        else:
            model.push(force_ind=0)
            model.push(force_ind=1)
        wait_for_shutdown(model)
    else:
        assert model(5.0) == 0.0
        model.pull()
        assert model(5.0) == 5.0
        wait_for_shutdown(model)

def test_multisource_fn(args):
    cluster_dict, module_name, task, task_idx, \
        source, sink, push_to_all = args

    clus = Cluster(cluster_dict, task, task_idx)
    clus.start()

    model = DistributedModule(DummyModel(),
                              cluster=clus,
                              data_source=source,
                              module_name=module_name,
                              task=task,
                              index=task_idx,
                              push_to_all=push_to_all)

    if task == source:
        model._base_model.v.assign(task_idx + 1)
        model.push()
        wait_for_shutdown(model)
    else:
        assert model(5.0) == 0.0
        model.pull()
        pull1 = model(5.0)
        assert pull1 == 5.0 or pull1 == 10.0
        model.pull()
        pull2 = model(5.0)
        if pull1 == 5.0:
            assert pull2 == 10.0
        else:
            assert pull2 == 5.0
        wait_for_shutdown(model)

def test_chain_fn(args):
    cluster_dict, module_name, task, task_idx, \
        source, sink, push_to_all = args

    clus = Cluster(cluster_dict, task, task_idx)
    clus.start()

    model = DistributedModule(DummyModel(),
                              cluster=clus,
                              data_source=source,
                              data_sink=sink,
                              module_name=module_name,
                              task=task,
                              index=task_idx,
                              push_to_all=push_to_all)
    if task == source:
        model._base_model.v.assign(1.0)
        model.push()
        wait_for_shutdown(model)
    elif task == module_name:
        model.pull()
        model.push()
        wait_for_shutdown(model)
    else:
        assert task == sink
        assert model(5.0) == 0.0
        model.pull()
        assert model(5.0) == 5.0
        wait_for_shutdown(model)

class TestDistributedModel(unittest.TestCase):

    def test_push(self):
        cluster_dict = {'learner': ['localhost:6006'],
                        'worker': ['localhost:6007', 'localhost:6008']}
        with multiprocessing.Pool(3) as pool:
            # cluster_dict, module_name, task, task_idx,
            # source, sink, push_to_all
            pool.map(testpush_fn, [(cluster_dict, 'worker', 'learner', 0,
                                     'learner', None, False),
                                    (cluster_dict, 'worker', 'worker', 0,
                                     'learner', None, False),
                                    (cluster_dict, 'worker', 'worker', 1,
                                     'learner', None, False)])
        with multiprocessing.Pool(3) as pool:
            pool.map(testpush_fn, [(cluster_dict, 'worker', 'learner', 0,
                                     'learner', None, True),
                                    (cluster_dict, 'worker', 'worker', 0,
                                     'learner', None, True),
                                    (cluster_dict, 'worker', 'worker', 1,
                                     'learner', None, True)])
        print("Done")

    def test_pull(self):
        cluster_dict = {'learner': ['localhost:6006'],
                        'worker': ['localhost:6007', 'localhost:6008']}
        with multiprocessing.Pool(3) as pool:
            # cluster_dict, module_name, task, task_idx,
            # source, sink, push_to_all
            pool.map(testpull_fn, [(cluster_dict, 'worker', 'learner', 0,
                                     'learner', None, False),
                                    (cluster_dict, 'worker', 'worker', 0,
                                     'learner', None, False),
                                    (cluster_dict, 'worker', 'worker', 1,
                                     'learner', None, False)])
        with multiprocessing.Pool(3) as pool:
            pool.map(testpull_fn, [(cluster_dict, 'worker', 'learner', 0,
                                     'learner', None, True),
                                    (cluster_dict, 'worker', 'worker', 0,
                                     'learner', None, True),
                                    (cluster_dict, 'worker', 'worker', 1,
                                     'learner', None, True)])
        print("Done")

    def test_multisource(self):
        cluster_dict = {'worker': ['localhost:6006'],
                        'learner': ['localhost:6007', 'localhost:6008']}
        with multiprocessing.Pool(3) as pool:
            # cluster_dict, module_name, task, task_idx,
            # source, sink, push_to_all
            pool.map(test_multisource_fn, [(cluster_dict, 'worker', 'learner', 0,
                                            'learner', None, False),
                                           (cluster_dict, 'worker', 'learner', 1,
                                            'learner', None, False),
                                           (cluster_dict, 'worker', 'worker', 0,
                                            'learner', None, False)])
        print("Done")

    def test_chain(self):
        cluster_dict = {'worker': ['localhost:6006'],
                        'learner': ['localhost:6007'],
                        'replay': ['localhost:6008']}
        with multiprocessing.Pool(3) as pool:
            # cluster_dict, module_name, task, task_idx,
            # source, sink, push_to_all
            pool.map(test_chain_fn, [(cluster_dict, 'replay', 'learner', 0,
                                            'worker', 'learner', False),
                                           (cluster_dict, 'replay', 'replay', 0,
                                            'worker', 'learner', False),
                                           (cluster_dict, 'replay', 'worker', 0,
                                            'worker', 'learner', False)])
        print("Done")


if __name__ == "__main__":
    unittest.main()
