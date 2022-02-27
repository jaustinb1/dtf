import tensorflow as tf

from dtf.modules import DistributedModel, Model, DistributedModule
from dtf.replay_buffer import get_replay_buffer, DistributedRelayBuffer
from dtf.environment import Env

class Learner:

    def __init__(self, model_cls, worker_spec, learner_spec,
                 cluster, task, task_index, replay_buffer_spec,
                 distributed=False):

        self._model_cls = model_cls
        self._cluster = cluster
        self._task = task
        self._task_index = task_index
        self._distributed = distributed
        self._learner_spec = learner_spec
        self._worker_spec = worker_spec
        self._replay_buffer_spec = replay_buffer_spec

        self._setup()

    def _setup(self):

        env = Env(self._worker_spec["env_name"], 1)

        self._model = self._model_cls(
            env.storage_spec
        )

        if self._distributed:
            self._model = DistributedModule(
                self._model, data_source="learner",
                data_sink=None, cluster=self._cluster,
                module_name="worker", task=self._task,
                index=self._task_index
            )
            self._replay_buffer = DistributedRelayBuffer(
                self._replay_buffer_spec["type"],
                env.storage_spec,
                self._replay_buffer_spec["capacity"],
                self._learner_spec["batch_size"],
                worker_name="worker",
                learner_name="learner",
                replay_buffer_name="replay_buffer",
                task_name=self._task,
                task_index=self._task_index,
                cluster=self._cluster,
            )
        else:
            self._replay_buffer = get_replay_buffer(
                self._replay_buffer_spec["type"])(
                    env.storage_spec,
                    self._replay_buffer_spec["capacity"]
                )

        env.close()
        del env

    def train_on_batch(self):
        raise NotImplementedError

    def run(self):

        self._model.push()
        while True:
            replay_data = self._replay_buffer.pull(return_data=True)

            self.train_on_batch(replay_data)

            self._model.push()
