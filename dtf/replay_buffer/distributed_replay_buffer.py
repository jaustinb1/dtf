from dtf.modules import DistributedModule
from dtf.replay_buffer import get_replay_buffer

from collections import namedtuple

Variable = namedtuple("Variable", ("shape", "name", "dtype"))

class DistributedRelayBuffer(DistributedModule):

    def __init__(self,
                 # replay buffer arguments
                 replay_buffer_type,
                 storage_spec,
                 capacity,
                 # output arguments
                 batch_size,
                 # distributed arguments
                 worker_name, learner_name,
                 replay_buffer_name, task_name,
                 task_index, cluster):

        self.batch_size = batch_size
        replay_buffer_cls = get_replay_buffer(replay_buffer_type)
        replay_buffer = replay_buffer_cls(storage_spec, capacity)
        super().__init__(replay_buffer,
                         cluster=cluster,
                         data_source=worker_name,
                         data_sink=learner_name,
                         module_name=replay_buffer_name,
                         task=task_name,
                         index=task_index,
                         push_to_all=False,
                         inbound_size=0,
                         outbound_size=batch_size)

    @property
    def distributed_variables(self):
        return [
            Variable(
                shape=v.shape[1:],
                name=v.name.split(":")[0],
                dtype=v.dtype) for v in self._base_model.variables]

    def run(self):

        while True:
            print("Waiting for data")
            self.pull()
            print("Got new data")
            replay_sample = self._base_model.sample_n(
                self.batch_size)
            self.push(data=replay_sample)
            print("Served a batch")
