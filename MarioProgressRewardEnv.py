import gym
import numpy as np
from bisect import bisect_left


class customRewardWrapper(gym.Wrapper):
    """
    Reward wrapper that heavily prioritizes distance gained:
    1. Massively rewards forward progress with high weight
    2. Gives enormous bonuses for reaching new max distances
    3. Provides huge milestone bonuses
    4. Severely penalizes standing still and going backwards
    5. Terminates episode if no progress is made for too long
    """

    def __init__(self, env, forward_weight=4, idle_penalty=0.4, time_penalty=0.02,
                 milestone_bonus=30.0, progress_timeout=32):
        super(customRewardWrapper, self).__init__(env)
        self.prev_x_pos = 0
        self.idle_counter = 0
        self.max_x_pos = 0
        self.forward_weight = forward_weight  # Significantly increased
        self.idle_penalty = idle_penalty
        self.time_penalty = time_penalty
        self.milestone_bonus = milestone_bonus  # Doubled
        self.flag_reached = False

        # Store milestones in a sorted list for efficient lookup
        # Add more fine-grained milestones for more frequent rewards
        self.milestones = sorted([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
                                  1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600,
                                  2800, 3000, 3200, 3400, 3600, 3800, 4000])
        self.next_milestone_idx = 0  # Track the next milestone to check

        # Progress timeout parameters
        self.progress_timeout = progress_timeout  # Max steps without forward progress
        self.steps_without_progress = 0
        self.last_progress_x_pos = 0

        # Pre-compute values
        self.idle_penalty_cap = 3.0
        self.new_territory_bonus = 5.0
        self.backward_penalty_multiplier = 3.0
        self.fixed_backward_penalty = 2.0
        self.fixed_idle_penalty = 0.5

        # Track highest distance achieved in episode
        self.episode_max_distance = 0

        # Distance achievements
        self.distance_achievements = {
            500: 500,
            725: 1000,
            1000: 2000,
            1500: 3500,
            1800: 4500,
            2200: 5500,
            2800: 7500,
            3000: 10000,
        }
        self.achieved_distances = set()

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.prev_x_pos = 0
        self.idle_counter = 0
        self.max_x_pos = 0
        self.next_milestone_idx = 0
        self.steps_without_progress = 0
        self.last_progress_x_pos = 0
        self.episode_max_distance = 0
        self.achieved_distances = set()
        self.flag_reached = False
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # Get current x position
        x_pos = info.get('x_pos', 0)

        # Calculate modified reward
        modified_reward = reward

        if info.get('flag_get', False) and not self.flag_reached and x_pos > 3100:
            print(
                f"REAL FLAG REACHED at distance {x_pos}! Applying massive completion bonus!")
            modified_reward += 10000.0
            self.flag_reached = True
            done = True

        # 1. Progress reward: heavily reward forward progress
        x_progress = x_pos - self.prev_x_pos

        if x_progress > 0:
            # Dramatically reward forward progress
            modified_reward += x_progress * self.forward_weight
            # Reset idle counter when moving
            self.idle_counter = 0
            # Update last progress position and reset timeout counter
            self.last_progress_x_pos = x_pos
            self.steps_without_progress = 0
        elif x_progress == 0:
            # Penalize standing still
            self.idle_counter += 1

            if self.idle_counter > 5:
                idle_factor = min(self.idle_counter / 5,
                                  self.idle_penalty_cap) ** 1.5
                modified_reward -= self.idle_penalty * idle_factor * 3.0

            modified_reward -= self.fixed_idle_penalty
            self.steps_without_progress += 1
        else:
            # Penalize going backward
            backward_penalty = abs(x_progress) * \
                self.backward_penalty_multiplier
            modified_reward -= backward_penalty
            modified_reward -= self.fixed_backward_penalty
            self.steps_without_progress += 1

        # 2. Time penalty: small penalty for each step
        modified_reward -= self.time_penalty

        # 3. New max position bonus: MASSIVE reward for reaching new furthest point
        if x_pos > self.max_x_pos:
            # Highly reward discovering new territory
            new_territory_gain = (x_pos - self.max_x_pos)
            modified_reward += new_territory_gain * self.new_territory_bonus
            self.max_x_pos = x_pos

        # Track episode max distance
        if x_pos > self.episode_max_distance:
            self.episode_max_distance = x_pos

            # Check for distance achievements
            for distance, bonus in self.distance_achievements.items():
                if distance not in self.achieved_distances and x_pos >= distance:
                    modified_reward += bonus
                    self.achieved_distances.add(distance)
                    print(
                        f"Achievement unlocked: Reached {distance} distance! +{bonus} reward")

        # 4. Milestone bonuses: frequent rewards for progress
        if self.next_milestone_idx < len(self.milestones) and x_pos >= self.milestones[self.next_milestone_idx]:
            while (self.next_milestone_idx < len(self.milestones) and
                   x_pos >= self.milestones[self.next_milestone_idx]):
                modified_reward += self.milestone_bonus
                self.next_milestone_idx += 1

        # 5. Progress timeout: terminate episode if stuck for too long
        if self.progress_timeout > 0 and self.steps_without_progress >= self.progress_timeout:
            print(
                f"Terminating episode: No progress for {self.steps_without_progress} steps")
            done = True
            modified_reward -= 100
            info['timeout'] = True

        # 6. Add final episode reward based on max distance if done
        if done:
            distance_reward = self.episode_max_distance * \
                0.2  # Additional reward based on max distance
            modified_reward += distance_reward
            print(
                f"Episode complete. Distance bonus: +{distance_reward:.1f} for reaching {self.episode_max_distance}")

        # Update previous position
        self.prev_x_pos = x_pos

        # Add custom info
        info['original_reward'] = reward
        info['modified_reward'] = modified_reward
        info['x_progress'] = x_progress
        info['idle_counter'] = self.idle_counter
        info['steps_without_progress'] = self.steps_without_progress
        info['max_distance'] = self.episode_max_distance

        return obs, modified_reward, done, info


# Keep the original class name for compatibility
MarioProgressRewardEnv = customRewardWrapper
