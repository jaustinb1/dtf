import gym
import mujoco_py

class GymEnv:

    def __init__(self, env_name, render_params=None):
        self.render_params = render_params
        self._env = gym.make(env_name)

        self._viewers = {}

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == "human":
                self.viewer = mujoco_py.MjViewer(
                    self._env.sim,
                    self.render_params["width"],
                    self.render_params["height"],
                    self.render_params["x"],
                    self.render_params["y"])
            elif mode == "rgb_array" or mode == "depth_array":
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)
            self.viewer._hide_overlay = True
            self._env.viewer = self.viewer
            self._env.unwrapped.viewer = self.viewer
            self.viewer_setup()
            self._viewers[mode] = self.viewer

        return self.viewer

    def render(self, mode="human"):
        return self._get_viewer(mode).render()

    def __getattr__(self, attr):
        return self._env.__getattr__(attr)

def make(env_name, render_params=None):
    return GymEnv(env_name, render_params)
