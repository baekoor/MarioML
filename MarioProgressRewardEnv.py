import gym
import numpy as np


class customRewardWrapper(gym.Wrapper):
    """
    Wrapper that enhances rewards to encourage progression through the level:
    1. Penalizes standing still (no x-position change)
    2. Rewards forward progress continuously
    3. Applies time penalty to encourage efficiency
    4. Gives extra reward for significant milestones
    5. Terminates episode if no progress is made for too long
    """

    def __init__(self, env, forward_weight=1.0, idle_penalty=0.1, time_penalty=0.01,
                 milestone_bonus=10.0, progress_timeout=150):
        super(customRewardWrapper, self).__init__(env)
        self.prev_x_pos = 0
        self.idle_counter = 0
        self.max_x_pos = 0
        self.forward_weight = forward_weight
        self.idle_penalty = idle_penalty
        self.time_penalty = time_penalty
        self.milestone_bonus = milestone_bonus
        self.milestones = [100, 200, 400, 800,
                           1200, 1600, 2000, 2400, 2800, 3200]
        self.achieved_milestones = set()

        # Progress timeout parameters
        self.progress_timeout = progress_timeout  # Max steps without forward progress
        self.steps_without_progress = 0
        self.last_progress_x_pos = 0

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.prev_x_pos = 0
        self.idle_counter = 0
        self.max_x_pos = 0
        self.achieved_milestones = set()
        self.steps_without_progress = 0
        self.last_progress_x_pos = 0
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # Get current x position
        x_pos = info.get('x_pos', 0)

        # Calculate modified reward
        modified_reward = reward

        # 1. Progress reward: reward for moving forward, more than default
        x_progress = x_pos - self.prev_x_pos
        if x_progress > 0:
            # Reward forward progress
            modified_reward += x_progress * self.forward_weight
            # Reset idle counter when moving
            self.idle_counter = 0
            # Update last progress position and reset timeout counter
            self.last_progress_x_pos = x_pos
            self.steps_without_progress = 0
        elif x_progress == 0:
            # Penalize standing still, with increasing penalty over time
            self.idle_counter += 1
            if self.idle_counter > 10:  # Allow a small buffer of idle frames
                modified_reward -= self.idle_penalty * \
                    min(self.idle_counter / 10, 3)  # Cap the penalty

            # Increment steps without progress
            self.steps_without_progress += 1
        else:
            # Going backward also counts as no progress
            self.steps_without_progress += 1

        # 2. Time penalty: small penalty for each step to encourage efficiency
        modified_reward -= self.time_penalty

        # 3. New max position bonus: extra reward for reaching new furthest point
        if x_pos > self.max_x_pos:
            modified_reward += (x_pos - self.max_x_pos) * \
                0.5  # Bonus for new territory
            self.max_x_pos = x_pos

        # 4. Milestone bonuses: significant one-time rewards for reaching distance milestones
        for milestone in self.milestones:
            if milestone not in self.achieved_milestones and x_pos >= milestone:
                modified_reward += self.milestone_bonus
                self.achieved_milestones.add(milestone)

        # 5. Progress timeout: terminate episode if stuck for too long
        if self.progress_timeout > 0 and self.steps_without_progress >= self.progress_timeout:
            print(
                f"Terminating episode: No progress for {self.steps_without_progress} steps")
            done = True
            modified_reward -= 50  # Large penalty for timing out
            info['timeout'] = True

        # Update previous position
        self.prev_x_pos = x_pos

        # Add custom info
        info['original_reward'] = reward
        info['modified_reward'] = modified_reward
        info['x_progress'] = x_progress
        info['idle_counter'] = self.idle_counter
        info['steps_without_progress'] = self.steps_without_progress

        return obs, modified_reward, done, info


# Keep the original class name for compatibility
MarioProgressRewardEnv = customRewardWrapper
