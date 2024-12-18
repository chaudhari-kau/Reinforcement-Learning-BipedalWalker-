import gym
from collections import deque
import numpy as np

class BoxToHistoryBox(gym.ObservationWrapper):
    '''
    This wrapper converts the environment which returns last h observations.
    First h observations are converted such that first states are same.
    '''
    def __init__(self, env, h=8):
        super().__init__(env)
        self.h = h
        self.obs_memory = deque(maxlen=self.h)
        shape = (h,) + self.observation_space.shape
        low = np.repeat(np.expand_dims(self.observation_space.low, 0), h, axis=0)
        high = np.repeat(np.expand_dims(self.observation_space.high, 0), h, axis=0)    
        self.observation_space = gym.spaces.Box(low, high, shape)

    def add_to_memory(self, obs):
        self.obs_memory.append(np.expand_dims(obs, axis=0))

    def observation(self, obs):
        self.add_to_memory(obs)
        return np.concatenate(self.obs_memory)

    def reset(self):
        reset_state = self.env.reset()
        for i in range(self.h-1):
            self.add_to_memory(reset_state)
        return self.observation(reset_state)

class MyWalker(gym.Wrapper):
    def __init__(self, env, skip=2):
        super().__init__(env)
        self._obs_buffer = deque(maxlen=skip)
        self._skip = skip
        # Update max steps based on environment
        self._max_episode_steps = 1000 if 'Hardcore' in env.spec.id else 800

    def step(self, action):
        total_reward = 0
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if self.env.game_over:
                reward = -100.0  # As per documentation for falling
                info["dead"] = True
            else:
                info["dead"] = False

            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
            info.update({
            'forward_reward': 0.0,  # Moving forward reward
            'energy_penalty': 0.0,  # Motor torque cost
            'fall_penalty': 0.0    # -100 for falling
        })
        
        return obs, total_reward, done, info

    def reset(self):
        return self.env.reset()

    def render(self, mode="human"):
        for _ in range(self._skip):
            out = self.env.render(mode=mode)
        return out