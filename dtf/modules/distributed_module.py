import tensorflow as tf
import time
import numpy as np

class DistributedModel(tf.Module):

    def __init__(self, update_method="assign"):
        super().__init__()
        self.update_method = update_method

    def get_scatter_idx(self):
        if self.update_method == "scatter":
            raise NotImplementedError
        return None

    def _do_update(self, var_idx, update, scatter_idx=None):
        if self.update_method == "assign":
            self.variables[var_idx].assign(update)
        elif self.update_method == "scatter":
            self.variables[var_idx] = tf.tensor_scatter_nd_update(
                self.variables[var_idx], [[scatter_idx]], update)
        else:
            raise NotImplementedError

    def update_variables(self, updates):
        scatter_index = self.get_scatter_idx()
        for i in range(len(self.variables)):
            name = self.variables[i].name
            self._do_update(i, updates[name], scatter_index)

    def handle_data(self, updates):
        self.update_variables(updates)


class DistributedModule:
    """
    A module which should communicate with other modules.
    A module is defined as a block which can optionally take
    as input (data_source) data output by another models, process
    it, and then optionally output to another module (data_sink).
    """
    def __init__(self, base_model,
                 data_source=None,
                 data_sink=None,
                 cluster=None,
                 module_name=None,
                 task=None,
                 index=None,
                 push_to_all=False,
                 inbound_size=0,
                 outbound_size=0):
        """
        base_model: the DistributedModel instance that we are wrapping
                    with communication
        data_source: an optional source of data that is inbound into this
                     module
        data_sink: an optional destination for outbound data from this module
        cluster: a dtf.cluster.Cluster object defining the cluster
        module_name: a key corresponding to a type of job in the cluster
                     this is the module name which can process the data
                     and serve as a relay between source and sink. For
                     instance this can be ReplayBuffer which takes as input
                     data from a worker and outputs data to a learner.
        task: the name corresponding to the current task, used to identify
              what job we are in the cluster
        index: the task index in the cluster corresponding to the current job
        push_to_all: if we are a data_source with multiple destination modules
                     or if we are an instance of module_name with multiple
                     data_sinks, a call to push() will push data to all
                     destinations.
        """
        self._base_model = base_model
        self.cluster = cluster
        self._inbound_size = inbound_size
        self._outbound_size = outbound_size

        self._data_source = data_source
        self._data_sink = data_sink
        self._module_name = module_name
        self._index = index
        self._push_to_all = push_to_all

        self._is_source = task == self._data_source
        self._is_sink = task == self._data_sink
        self._is_module = task == self._module_name

        self._num_module = cluster.count(module_name)
        self._num_sources = 0 if not data_source else cluster.count(data_source)
        self._num_sinks = 0 if not data_sink else cluster.count(data_sink)

        self._update_queues = {}

        self._make_queues()

    @property
    def variables(self):
        return self._base_model.variables

    @property
    def distributed_variables(self):
        return self.variables

    def __call__(self, x):
        return self._base_model(x)

    def _make_queues(self):

        variables = self.distributed_variables

        dtypes = [v.dtype for v in variables]
        shapes = [v.shape for v in variables]
        names =  [v.name for v in variables]

        inbound_shapes = []
        outbound_shapes = []
        for i in range(len(shapes)):
            if self._inbound_size > 0:
                inbound_shapes.append([self._inbound_size] + shapes[i])
            else:
                inbound_shapes.append(shapes[i])
            if self._outbound_size > 0:
                outbound_shapes.append([self._outbound_size] + shapes[i])
            else:
                outbound_shapes.append(shapes[i])


        # By convention, we set the device for a queue to be
        # on the source of the queue rather than the destination

        # Build the inbound queues
        inbound_queues = []
        inbound_devices = []
        prefix = f"{self._module_name}InboundFrom{self._data_source}"
        for source_ind in range(self._num_sources):
            queue_buff = []
            devices_buff = []
            for module_ind in range(self._num_module):
                queue_buff.append(
                    f"{prefix}({source_ind},{module_ind})")
                devices_buff.append(
                    self.cluster.get_device(self._data_source, source_ind)
                )

            inbound_queues.append(queue_buff)
            inbound_devices.append(devices_buff)

        # Build the outbound queues
        outbound_queues = []
        outbound_devices = []
        prefix = f"{self._module_name}OutboundTo{self._data_sink}"
        for module_ind in range(self._num_module):
            queue_buff = []
            devices_buff = []
            for sink_ind in range(self._num_sinks):
                queue_buff.append(
                    f"{prefix}({module_ind},{sink_ind})")
                devices_buff.append(
                    self.cluster.get_device(
                        self._module_name, module_ind))
            outbound_queues.append(queue_buff)
            outbound_devices.append(devices_buff)
        for source_ind in range(self._num_sources):
            for module_ind in range(self._num_module):
                with tf.device(inbound_devices[source_ind][module_ind]):
                    name = inbound_queues[source_ind][module_ind]
                    self._update_queues[name] = tf.queue.FIFOQueue(
                        capacity=10, dtypes=dtypes,
                        shapes=inbound_shapes, names=names,
                        shared_name=name,
                        name=name)
        for module_ind in range(self._num_module):
            for sink_ind in range(self._num_sinks):
                with tf.device(outbound_devices[module_ind][sink_ind]):
                    name = outbound_queues[module_ind][sink_ind]
                    self._update_queues[name] = tf.queue.FIFOQueue(
                        capacity=10, dtypes=dtypes,
                        shapes=outbound_shapes, names=names,
                        shared_name=name,
                        name=name)

    def push(self, data=None, force_ind=None):

        assert self._is_source or self._is_module
        assert (self._is_source and self._num_sources) or (
            self._is_module and self._num_module)
        assert not (self._push_to_all and (force_ind is not None))

        num_sinks = self._num_module if self._is_source else self._num_sinks
        if force_ind is not None:
            sink_inds = [force_ind]
        elif not self._push_to_all:
            sink_inds = [
                np.random.choice(
                    range(num_sinks))
            ]
        else:
            # push to all
            sink_inds = list(range(num_sinks))

        if self._is_source:
            prefix = f"{self._module_name}InboundFrom{self._data_source}"
        else:
            prefix = f"{self._module_name}OutboundTo{self._data_sink}"

        num_sources = self._num_sources if self._is_source else self._num_module
        for source_ind in range(num_sources):
            if source_ind != self._index:
                continue
            for sink_ind in sink_inds:
                name = f"{prefix}({source_ind},{sink_ind})"
                if data:
                    # If we have data, we send it
                    self._update_queues[name].enqueue({
                        k: v for k, v in data.items()
                    })
                else:
                    # If we do not have data, assume we are sending
                    # the variables from the current module
                    self._update_queues[name].enqueue({
                        v.name: v for v in self.variables
                    })
                print("Pushed updates")

    def pull(self, wait=True, return_data=False):
        assert self._is_sink or self._is_module
        assert (self._is_sink and self._num_sinks) or (
            self._is_module and self._num_module)

        if self._is_module:
            prefix = f"{self._module_name}InboundFrom{self._data_source}"
        else:
            prefix = f"{self._module_name}OutboundTo{self._data_sink}"

        num_source = self._num_module if self._is_sink else self._num_sources
        num_sink = self._num_module if self._is_module else self._num_sinks

        def avaialble_updates():
            return [self._update_queues[f"{prefix}({s},{self._index})"].size() for s in
                    range(num_source)]
        if wait:
            while np.sum(avaialble_updates()) == 0:
                print("waiting for update")
                time.sleep(1)

        for source_ind in range(num_source):
            for sink_ind in range(num_sink):
                if sink_ind != self._index:
                    continue
                name = f"{prefix}({source_ind},{sink_ind})"
                if self._update_queues[name].size() > 0:
                    update = self._update_queues[name].dequeue()
                    print("Pulled update")
                    if self._is_sink and return_data:
                        return update
                    self._base_model.handle_data(update)
                    return
        print("No updates")
        return None

    def maybe_update(self):
        return self.pull(wait=False)
