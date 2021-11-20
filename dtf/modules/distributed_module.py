import tensorflow as tf
import time

class DistributedModel:
    def __init__(self, base_model,
                 cluster=None,
                 source=None,
                 sink=None,
                 mode=None,
                 index=None,
                 multi_sink=False,
                 storage_spec=None):
        self._base_model = base_model
        self._cluster = cluster
        self._multi_sink = multi_sink
        self._storage_spec = storage_spec

        if multi_sink:
            # If we are a multisink, we assume only one source
            self.update_device = self._cluster.get_device(source)
            self._num_replicas = cluster.count(sink)
        else:
            # If we are not multisink, we are multisource and index is
            # the source index.
            self.update_device = self._cluster.get_device(sink)
            self._num_replicas = cluster.count(source)

        self.update_name = f"{source}->{sink}"

        self.is_source = mode == source

        self.index = index

        self._update_queues = {}
        self._make_update()

    def __call__(self, x):
        return self._base_model(x)

    @property
    def variables(self):
        return self._base_model.variables

    def _make_update(self):
        with tf.device(self.update_device):
            if self._multi_sink:
                variables = self.variables
                dtypes = [v.dtype for v in variables]
                shapes = [v.shape for v in variables]
                names = [v.name for v in variables]
            else:
                dtypes = self._storage_spec['dtypes']
                shapes = self._storage_spec['shapes']
                names = self._storage_spec['names']

            tag = [f"({i})" for i in range(self._num_replicas)]
            if self._multi_sink:
                name_fn = lambda x: self.update_name + x
            else:
                name_fn = lambda x: x + self.update_name

            for s in tag:
                self._update_queues[s] = tf.queue.FIFOQueue(
                    capacity=1, dtypes=dtypes,
                    shapes=shapes, names=names,
                    shared_name=name_fn(s),
                    name=name_fn(s))

    def push(self, data=None):
        assert self.is_source

        for i in range(self._num_replicas):
            s = f"({i})"
            if not self._multi_sink and i != self.index:
                continue
            if data:
                self._update_queues[s].enqueue({
                    k: v for k, v in data.items()
                })
            else:
                self._update_queues[s].enqueue({
                    v.name: v for v in self.variables
                })
        print("Pushed Updates")

    def maybe_pull(self):
        assert not self.is_source
        s = f"({self.index})"
        if self._update_queues[s].size() > 0:
            update = self._update_queues[s].dequeue()
            for i in range(len(self.variables)):
                name = self.variables[i].name
                self.variables[i].assign(update[name])
            print("Successfully pulled update")
        else:
            print("Nothing to pull")

    def pull(self):
        assert not self.is_source
        s = f"({self.index})"
        while self._update_queues[s].size() == 0:
            print("waiting for update")
            time.sleep(1)
        update = self._update_queues[s].dequeue()
        for i in range(len(self.variables)):
            name = self.variables[i].name
            self.variables[i].assign(update[name])
        print("Successfully pulled update")
