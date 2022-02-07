import tensorflow as tf
import time
import numpy as np

class DistributedModel(tf.Module):

    def __init__(self, update_method="assign"):
        super().__init__()
        self.update_method = update_method

    def get_scatter_idx(self, var_name):
        raise NotImplementedError

    def _do_update(self, var_idx, update):
        if self.update_method == "assign":
            self.variables[var_idx].assign(update)
        elif self.update_method == "scatter":
            idx = self.get_scatter_index(self.variables[var_idx].name)
            self.variables[var_idx][idx] = update
        else:
            raise NotImplementedError

    def update_variables(self, updates):
        for i in range(len(self.variables)):
            name = self.variables[i].name
            self._do_update(i, updates[name])


class DistributedModule:
    def __init__(self, base_model,
                 cluster=None,
                 source=None,
                 sink=None,
                 mode=None,
                 index=None,
                 storage_spec=None,
                 push_to_all=False):
        self._base_model = base_model
        self._cluster = cluster
        self._multi_sink = cluster.count(sink) > 1
        self._multi_source = cluster.count(source) > 1
        self._storage_spec = storage_spec
        self.is_source = mode == source
        self.source = source
        self.sink = sink
        self._push_to_all = push_to_all
        self._matrixed = self._multi_sink and self._multi_source

        if self._multi_sink and self._multi_source:
            self._num_replicas = cluster.count(sink) * cluster.count(source)
            self.update_devices = []
            for _ in range(cluster.count(source)):
                buff = []
                for sink_ind in range(cluster.count(sink)):
                    buff.append(self._cluster.get_device(sink, sink_ind))
                self.update_devices.append(buff)
        elif self._multi_sink:
            # Single source, multiple sinks
            assert not self._multi_source
            self._num_replicas = cluster.count(sink)
            self.update_devices = [[
                self._cluster.get_device(source)] * self._num_replicas]
        else:
            # Multiple sources, single sink
            assert not self._multi_sink
            self._num_replicas = cluster.count(source)
            self.update_devices = [
                [self._cluster.get_device(sink)]]*self._num_replicas

        self.update_name = f"{source}->{sink}"
        self.index = index

        self._update_queues = {}
        self._make_update()

    def __call__(self, x):
        return self._base_model(x)

    @property
    def variables(self):
        return self._base_model.variables

    def _make_update(self):
        if self._multi_sink and not self._matrixed:
            variables = self.variables
            dtypes = [v.dtype for v in variables]
            shapes = [v.shape for v in variables]
            names = [v.name for v in variables]
        else:
            dtypes = self._storage_spec['dtypes']
            shapes = self._storage_spec['shapes']
            names = self._storage_spec['names']

        tag = [f"({i})" for i in range(self._num_replicas)]
        tag = []
        for source_ind in range(self._cluster.count(self.source)):
            buff = []
            for sink_ind in range(self._cluster.count(self.sink)):
                buff.append(f"({source_ind},{sink_ind})")
            tag.append(buff)

        if self._multi_sink:
            name_fn = lambda x: self.update_name + x
        else:
            name_fn = lambda x: x + self.update_name

        for source_ind in range(len(tag)):
            for sink_ind in range(len(tag[source_ind])):
                with tf.device(self.update_devices[source_ind][sink_ind]):
                    s = tag[source_ind][sink_ind]
                    self._update_queues[s] = tf.queue.FIFOQueue(
                        capacity=1, dtypes=dtypes,
                        shapes=shapes, names=names,
                        shared_name=name_fn(s),
                        name=name_fn(s))

    def push(self, data=None, force_ind=None):
        assert self.is_source

        if not self._multi_sink or self._push_to_all:
            # single sink or multi sink but pushing updates to all
            sink_id = range(self._cluster.count(self.sink))
        else:
            assert not self._push_to_all
            if force_ind is not None:
                sink_id = [force_ind]
            else:
                sink_id = [np.random.choice(
                    range(self._cluster.count(self.sink)))]
        for source_ind in range(self._cluster.count(self.source)):
            if source_ind != self.index:
                continue
            for sink_ind in sink_id:
                s = f"({source_ind},{sink_ind})"
                if data:
                    self._update_queues[s].enqueue({
                        k: v for k, v in data.items()
                    })
                else:
                    self._update_queues[s].enqueue({
                        v.name: v for v in self.variables
                    })
        print("Pushed all updates")

    def maybe_pull(self):
        assert not self.is_source

        for source_ind in range(self._cluster.count(self.source)):
            for sink_ind in range(self._cluster.count(self.sink)):
                if sink_ind != self.index:
                    continue
                s = f"({source_ind},{sink_ind})"
                if self._update_queues[s].size() > 0:
                    update = self._update_queues[s].dequeue()
                    self._base_model.update_variables(update)
                    print("Successfully pulled update")
                    return
                else:
                    print("Nothing to pull")

    def pull(self):
        assert not self.is_source

        def avaialble_updates():
            return [self._update_queues[f"({s},{self.index})"].size() for s in
                    range(self._cluster.count(self.source))]

        while np.sum(avaialble_updates()) == 0:
            print("waiting for update")
            time.sleep(1)

        updates = avaialble_updates()
        for i in range(len(updates)):
            if updates[i] > 0:
                s = f"({i},{self.index})"
                update = self._update_queues[s].dequeue()
                self._base_model.update_variables(update)
                print("Successfully pulled update")
                break
