import unittest
import tensorflow as tf
import multiprocessing
import time
from dtf.cluster import Cluster
from dtf.modules import DistributedModel
import numpy as np

physical_devices = tf.config.list_physical_devices('GPU')
try:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass


class DummyModel(tf.Module):
    def __init__(self):
        super().__init__()
        self.v = tf.Variable(0.0)

    def __call__(self, x):
        return x * self.v

def wait_for_shutdown(distributed_model):
    while True:
        time.sleep(1)
        flg = True
        for q in distributed_model._update_queues:
            try:
                if distributed_model._update_queues[q].size() > 0:
                    flg = False
                    break
            except:
                # workers have finished
                return
        if flg:
            break

def multisink_fn(args):
    cluster_dict, name, task, source, sink, push_to_all = args

    clus = Cluster(cluster_dict, name, task)
    clus.start()

    model = DistributedModel(DummyModel(),
                             cluster=clus,
                             source=source,
                             sink=sink,
                             mode=name,
                             index=task, push_to_all=push_to_all)
    if name == source:
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

def multisource_fn(args):
    cluster_dict, name, task, source, sink = args

    clus = Cluster(cluster_dict, name, task)
    clus.start()

    model = DistributedModel(DummyModel(),
                             cluster=clus,
                             source=source,
                             sink=sink,
                             mode=name,
                             index=task,
                             storage_spec={'dtypes': [tf.float32],
                                           'shapes': [(1,)],
                                           'names': ['test']})
    if name == source:
        model.push({'test': np.array([task])})
        wait_for_shutdown(model)
    else:
        found_updates = []
        while True:
            for s in model._update_queues:
                if model._update_queues[s].size() > 0:
                    update = model._update_queues[s].dequeue()
                    assert 'test' in update
                    found_updates.append(update['test'].numpy()[0])
                    print("Found update")
            if len(found_updates) == clus.count(source):
                break
            time.sleep(1)
        wait_for_shutdown(model)

        assert 0 in found_updates
        assert 1 in found_updates

def matrixed_fn(args):
    cluster_dict, name, task, source, sink = args

    clus = Cluster(cluster_dict, name, task)
    clus.start()

    model = DistributedModel(DummyModel(),
                             cluster=clus,
                             source=source,
                             sink=sink,
                             mode=name,
                             index=task,
                             storage_spec={'dtypes': [tf.float32],
                                           'shapes': [(1,)],
                                           'names': ['test']},
                             push_to_all=True)
    if name == source:
        model.push({'test': np.array([task])})
        wait_for_shutdown(model)
    else:
        found_updates = []
        while True:
            for s in model._update_queues:
                if int(s[-2]) != task:
                    continue
                try:
                    if model._update_queues[s].size() > 0:
                        update = model._update_queues[s].dequeue()
                        assert 'test' in update
                        found_updates.append(update['test'].numpy()[0])
                        print("Found update", task, found_updates)
                except Exception as e:
                    print(e)
                    break
            if len(found_updates) == clus.count(source):
                break
            time.sleep(1)
        wait_for_shutdown(model)
        print(found_updates)
        assert 0 in found_updates
        assert 1 in found_updates



class TestDistributedModel(unittest.TestCase):

    def test_multisink_push_to_one(self):
        cluster_dict = {'learner': ['localhost:6006'],
                        'worker': ['localhost:6007', 'localhost:6008']}
        with multiprocessing.Pool(3) as pool:
            pool.map(multisink_fn, [(cluster_dict, 'learner', 0,
                                     'learner', 'worker', False),
                                    (cluster_dict, 'worker', 0,
                                     'learner', 'worker', False),
                                    (cluster_dict, 'worker', 1,
                                     'learner', 'worker', False)])
        print("Done")

    def test_multisink_push_to_all(self):
        cluster_dict = {'learner': ['localhost:6006'],
                        'worker': ['localhost:6007', 'localhost:6008']}
        with multiprocessing.Pool(3) as pool:
            pool.map(multisink_fn, [(cluster_dict, 'learner', 0,
                                     'learner', 'worker', True),
                                    (cluster_dict, 'worker', 0,
                                     'learner', 'worker', True),
                                    (cluster_dict, 'worker', 1,
                                     'learner', 'worker', True)])
        print("Done")

    def test_multisource(self):
        cluster_dict = {'learner': ['localhost:6006'],
                        'worker': ['localhost:6007', 'localhost:6008']}
        with multiprocessing.Pool(3) as pool:
            pool.map(multisource_fn, [(cluster_dict, 'learner', 0,
                                     'worker', 'learner'),
                                    (cluster_dict, 'worker', 0,
                                     'worker', 'learner'),
                                    (cluster_dict, 'worker', 1,
                                     'worker', 'learner')])
        print("Done")

    def test_matrixed(self):
        cluster_dict = {'learner': ['localhost:6005', 'localhost:6006'],
                        'worker': ['localhost:6007', 'localhost:6008']}
        with multiprocessing.Pool(4) as pool:
            pool.map(matrixed_fn, [(cluster_dict, 'learner', 0,
                                     'worker', 'learner'),
                                      (cluster_dict, 'learner', 1,
                                     'worker', 'learner'),
                                      (cluster_dict, 'worker', 0,
                                       'worker', 'learner'),
                                      (cluster_dict, 'worker', 1,
                                       'worker', 'learner')])
        print("Done")


if __name__ == "__main__":
    unittest.main()
