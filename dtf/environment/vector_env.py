import multiprocessing as mp
import numpy as np
import glfw
from dtf.environment import gym_shm
import cv2
import json
import random
from collections import defaultdict, deque


class Env:

    def __init__(self, env_name, num_envs):
        self.num_envs = num_envs
        self.env_name = env_name

        self._procs = []
        self._pipes = []

        self._setup_envs(env_name, num_envs)

    def _setup_envs(self, name, num):

        self.proto_env = gym_shm.make(name)

        for i in range(num):
            pipe = mp.Pipe()
            proc = mp.Process(target=self._env_worker,
                              args=(pipe, i))
            self._procs.append(proc)
            self._pipes.append(pipe[0])

            self._procs[-1].daemon = True
            self._procs[-1].start()

    def _send(self, cmd, data=None, only_one=False):
        if data is None:
            data = [None] * self.num_envs
        if isinstance(data, str):
            data = [data] * self.num_envs

        for i in range(self.num_envs):
            self._pipes[i].send((cmd, data[i]))
            if only_one:
                break

    def _recv(self, only_one=False):
        ret_data = defaultdict(lambda: [])
        bound = 1 if only_one else self.num_envs
        for i in range(bound):
            data = self._pipes[i].recv()
            if data is None:
                continue
            for k, v in data.items():
                ret_data[k].append(v)

        for k in ret_data:
            ret_data[k] = np.stack(ret_data[k], axis=0)
        return ret_data

    def _get_position_from_i(self, ind):
        DOCK_SIZE = 80
        TOP_BAR_SIZE = 25
        WINDOW_BAR_SIZE = 28

        glfw.init()
        resolution, _, _ = glfw.get_video_mode(glfw.get_primary_monitor())

        width, height = resolution

        dim_in_sims_horiz = np.ceil(np.sqrt(self.num_envs))
        dim_in_sims_vert = np.ceil(self.num_envs / dim_in_sims_horiz)

        row, col = ind // dim_in_sims_horiz, ind % dim_in_sims_horiz

        sim_width = (width - DOCK_SIZE) // dim_in_sims_horiz
        sim_height = (height - dim_in_sims_vert*WINDOW_BAR_SIZE-TOP_BAR_SIZE) // dim_in_sims_vert

        return {
            "width": sim_width,
            "height": sim_height,
            "x": col * sim_width + DOCK_SIZE,
            "y": row * sim_height + row*WINDOW_BAR_SIZE + TOP_BAR_SIZE,
        }

    def _env_worker(self, pipe, ind):
        pipe[0].close()
        pipe = pipe[1]

        render_params = self._get_position_from_i(ind)

        np.random.seed(ind*100)
        random.seed(ind*100)

        env = gym_shm.make(self.env_name,
                           render_params=render_params)
        env.seed(ind*100)

        rewards_buff = deque(maxlen=1)
        rewards_list = []
        done = False
        obs = None
        while True:
            mode, data = pipe.recv()
            return_data = None
            if mode == "close":
                env.close()
                return
            if mode == "seed":
                np.random.seed(data)
                random.seed(data)
                env.seed(data)
            elif mode == "render":
                res  = env.render(mode=data)
                if data == "rgb_array":
                    return_data = {"image": res}
            elif mode == "step":
                obs, rew, done, info = env.step(data)
                rewards_list.append(rew)

                return_data = {
                    "observation": obs.copy(),
                    "reward": rew,
                    "done": done,
                    "info": info
                }
                if done:
                    obs = env.reset()
                    rewards_buff.append(rewards_list)
                    rewards_list = []
            elif mode == "get_obs":
                if obs is None:
                    obs = env.reset()
                return_data = {"observation": obs}
            elif mode == "reset":
                obs = env.reset()
                return_data = {"observation": obs}

                rewards_buff.append(rewards_list)
                rewards_list = []
            elif mode == "reward":
                return_data = {"reward": list(rewards_buff)}

            pipe.send(return_data)

    @property
    def action_space(self):
        return self.proto_env.action_space

    @property
    def observation_space(self):
        return self.proto_env.abservation_space

    def reset(self):
        self._send("reset")
        return self._recv()

    def get_obs(self):
        self._send("obs")
        return self._recv()

    def avg_return(self):
        self._send("reward")
        data = self._recv()
        return np.mean(np.sum(data, axis=-1))

    def max_step_reward(self, is_eval=False):
        self._send("reward")
        data = self._recv()
        return np.max(data)

    def step(self, action):
        self._send("step", action)
        return self._recv()

    def render(self, mode='human', only_one=False):
        self._send("render", mode, only_one=only_one)
        return self._recv(only_one=only_one)

    def close(self):
        self._send("close")
        for i in range(self.num_envs):
            self._procs[i].join()
