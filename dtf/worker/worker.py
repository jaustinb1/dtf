import tensorflow as tf

from dtf.environment import Env
from dtf.modules import DistributedModel, Model, DistributedModule
from dtf.replay_buffer import get_replay_buffer, DistributedRelayBuffer

class Worker:

    def __init__(self, env_name, num_envs,
                 model_cls, replay_buffer_spec,
                 learner_spec,
                 cluster=None, task=None,
                 task_index=None, distributed=False):
        self._env_name = env_name
        self._num_envs = num_envs
        self._cluster = cluster
        self._task = task
        self._task_index = task_index
        self._distributed = distributed

        self._model_cls = model_cls
        self._replay_buffer_spec = replay_buffer_spec
        self._learner_spec = learner_spec

        self._setup()

    def _setup(self):
        self._env = Env(self._env_name, self._num_envs)

        self._model = self._model_cls(
            self._env.storage_spec)

        if self._distributed:
            self._model = DistributedModule(
                self._model, data_source="learner",
                data_sink=None, cluster=self._cluster,
                module_name="worker", task=self._task,
                index=self._task_index,
            )
            self._replay_buffer = DistributedRelayBuffer(
                self._replay_buffer_spec["type"],
                self._env.storage_spec,
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
                    self._env.storage_spec,
                    self._replay_buffer_spec["capacity"]
                )

    def run(self):

        self._model.pull()
        while True:
            self._env.reset()
            done = False
            while not done:
                obs = self._env.get_obs()
                action = self._model(obs, explore=True)
                replay_data = self._env.step(action.numpy())
                done = replay_data["done"]

                self._replay_buffer.push(replay_data)

            self._model.maybe_pull()
