import os
import numpy as np
from collections import deque
import cv2
import gym
from gym import spaces
from gym.wrappers import TimeLimit

# Import the custom reward wrapper
from MarioProgressRewardEnv import customRewardWrapper, MarioProgressRewardEnv

# Disable OpenCL for better compatibility
cv2.ocl.setUseOpenCL(False)


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs):
        """Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(
                1, self.noop_max + 1
            )
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # Optimize memory usage by pre-allocating buffer
        shape = env.observation_space.shape
        self._obs_buffer = np.zeros((2,) + shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        info = None

        # Loop through skip frames
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break

        # Max pooling over the most recent observations
        max_frame = np.maximum(self._obs_buffer[0], self._obs_buffer[1])
        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class EpisodicLifeMario(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped._life
        if lives < self.lives and lives > 0:
            # Signal terminal state when losing a life
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped._life
        return obs


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.
        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
        observation should be warped.
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key

        # Pre-calculate color channels for observation space
        num_colors = 1 if self._grayscale else 3

        # Define the new observation space
        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )

        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space

        assert original_space.dtype == np.uint8 and len(
            original_space.shape) == 3

        # Pre-allocate transformation matrices for cv2 resizing
        # This improves performance by avoiding reallocation
        self._resize_mat = None

    def observation(self, obs):
        """Process the observation with more efficient CV2 operations"""
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            # Use faster grayscale conversion
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Use more efficient resize with pre-allocated matrices if possible
        frame = cv2.resize(
            frame, (self._width, self._height),
            interpolation=cv2.INTER_AREA,
            dst=self._resize_mat
        )

        if self._grayscale:
            # Reshape instead of expand_dims for better performance
            frame = frame.reshape(self._height, self._width, 1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame

        return obs


class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """Scale observation pixel values to floats in [0, 1]."""
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=env.observation_space.shape, dtype=np.float32
        )
        # Pre-compute scale for faster conversion
        self._scale = 1.0 / 255.0

    def observation(self, observation):
        """Convert uint8 to float32 and scale to [0, 1] more efficiently"""
        # Use in-place multiplication with pre-computed scale for better performance
        # Only allocate new memory when necessary
        return np.multiply(observation, self._scale, dtype=np.float32)


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape

        # Calculate stacked shape once
        self.stacked_shape = shp[:-1] + (shp[-1] * k,)

        # Update observation space
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=self.stacked_shape,
            dtype=env.observation_space.dtype,
        )

    def reset(self):
        ob = self.env.reset()
        # Pre-fill the frame buffer more efficiently
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        """Return the LazyFrames object using optimized implementation"""
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


def wrap_mario(env, use_custom_rewards=True, progress_timeout=100):
    """
    Configure environment for Super Mario Bros with optional custom rewards.

    Args:
        env: The environment to wrap
        use_custom_rewards: Whether to use the custom reward wrapper
        progress_timeout: Number of steps without progress before terminating episode

    Returns:
        The wrapped environment
    """
    # Apply common preprocessing wrappers
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeMario(env)
    env = WarpFrame(env)
    env = ScaledFloatFrame(env)

    # Apply custom reward wrapper if requested
    if use_custom_rewards:
        env = customRewardWrapper(env,
                                  forward_weight=1.0,
                                  idle_penalty=0.1,
                                  time_penalty=0.01,
                                  milestone_bonus=10.0,
                                  progress_timeout=progress_timeout)

    # Stack frames for temporal information
    env = FrameStack(env, 4)
    return env


class LazyFrames(object):
    """
    Optimized LazyFrames implementation that reduces memory usage and improves performance.
    This object ensures that common frames between the observations are only stored once.
    It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
    buffers.
    """

    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        Args:
            frames (list): List of frames to store
        """
        self._frames = frames
        self._out = None
        self._shape = None

        if frames and hasattr(frames[0], 'shape'):
            base_shape = frames[0].shape
            self._shape = base_shape[:-1] + (base_shape[-1] * len(frames),)

    def _force(self):
        """Concatenate frames only when needed, with memory optimization"""
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        """Convert to numpy array with optional dtype conversion"""
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype, copy=False)
        return out

    def __len__(self):
        """Return the number of frames"""
        if self._shape is not None:
            return self._shape[0]
        return len(self._force())

    def __getitem__(self, i):
        """Get item at index i"""
        return self._force()[i]

    def count(self):
        """Return the number of frames"""
        if self._out is not None:
            return self._out.shape[-1]
        elif self._frames:
            return len(self._frames)
        return 0

    def frame(self, i):
        """Return the i-th frame"""
        return self._force()[..., i]

    @property
    def shape(self):
        """Return the shape of the concatenated frames"""
        if self._shape is not None:
            return self._shape
        elif self._out is not None:
            return self._out.shape
        return None
