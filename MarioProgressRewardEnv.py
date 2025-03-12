import gym
import numpy as np
from bisect import bisect_left


class customRewardWrapper(gym.Wrapper):
    """
    Enhanced reward wrapper with extreme focus on forward progress and time optimization
    once max distance is reached:
    1. Massive reward for forward progress (highest priority)
    2. Extreme penalties for going backwards
    3. Time optimization once max distance (3155) is approached
    4. Quick termination if agent gets stuck or repeatedly goes backwards
    5. Anti-wall behavior penalties to prevent getting stuck at walls
    """

    def __init__(self, env, forward_weight=10, idle_penalty=0.8, time_penalty=0.05,
                 milestone_bonus=40.0, progress_timeout=25, max_distance=3155):
        super(customRewardWrapper, self).__init__(env)
        self.prev_x_pos = 0
        self.idle_counter = 0
        self.max_x_pos = 0
        self.forward_weight = forward_weight  # Increased further
        self.idle_penalty = idle_penalty  # Doubled
        self.time_penalty = time_penalty  # Increased
        self.milestone_bonus = milestone_bonus  # Increased
        self.flag_reached = False
        self.max_game_distance = max_distance  # Max possible distance in the level

        # Track actions to detect running into walls
        self.last_actions = []
        self.max_action_history = 10
        self.wall_counter = 0
        # How many same-direction actions to consider "stuck at wall"
        self.same_direction_threshold = 8

        # Store milestones in a sorted list for efficient lookup
        # Add more fine-grained milestones for more frequent rewards
        self.milestones = sorted([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
                                  1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600,
                                  2800, 3000, 3155])
        self.next_milestone_idx = 0  # Track the next milestone to check

        # Progress timeout parameters
        # Reduced for quicker termination of stuck episodes
        self.progress_timeout = progress_timeout
        self.steps_without_progress = 0
        self.last_progress_x_pos = 0
        self.backward_counter = 0
        self.max_backward_steps = 15  # If going backward for this many steps, terminate

        # Pre-compute values
        self.idle_penalty_cap = 5.0  # Increased cap
        self.new_territory_bonus = 8.0  # Increased bonus for new territory
        self.backward_penalty_multiplier = 5.0  # Much stronger backward penalty
        self.fixed_backward_penalty = 3.5  # Increased
        self.fixed_idle_penalty = 1.0  # Doubled
        self.time_optimization_threshold = 3000  # When to start optimizing for time

        # Track highest distance achieved in episode
        self.episode_max_distance = 0

        # Distance achievements
        self.distance_achievements = {
            500: 500,
            725: 1000,
            900: 1500,
            1000: 2500,
            1500: 3500,
            1800: 4500,
            2000: 5000,
            2200: 5500,
            2800: 7500,
            3000: 10000,
            3100: 15000,
        }
        self.achieved_distances = set()

        # Time optimization bonus once you reach near max distance
        self.near_max_distance_reached = False
        self.time_optimization_bonus = 20.0
        self.level_completion_time = 0  # Track time for completion speed bonus

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
        self.last_actions = []
        self.wall_counter = 0
        self.backward_counter = 0
        self.near_max_distance_reached = False
        self.level_completion_time = 0
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.level_completion_time += 1

        # Track actions to detect wall behavior
        self.last_actions.append(action)
        if len(self.last_actions) > self.max_action_history:
            self.last_actions.pop(0)

        # Get current x position
        x_pos = info.get('x_pos', 0)

        # Detect unrealistic position changes
        if x_pos > 5000 or x_pos < 0:
            print(
                f"WARNING: Suspicious x_pos value: {x_pos}. Previous x_pos: {self.prev_x_pos}")
            # Cap the position to something reasonable
            x_pos = min(max(0, self.prev_x_pos + 5), 5000)
            info['x_pos'] = x_pos
            print(f"Corrected x_pos to: {x_pos}")

        # Calculate modified reward
        modified_reward = reward

        # Check for true flag completion
        if info.get('flag_get', False) and not self.flag_reached and x_pos > 3100:
            # Faster completion = bigger bonus
            speed_bonus = max(0, 30000 - (self.level_completion_time * 5))
            print(
                f"REAL FLAG REACHED at distance {x_pos} in {self.level_completion_time} steps! Speed bonus: {speed_bonus}")
            modified_reward += 15000.0 + speed_bonus
            self.flag_reached = True
            done = True

        # 1. Progress reward: extremely reward forward progress
        x_progress = x_pos - self.prev_x_pos

        if x_progress > 0:
            # Dramatically reward forward progress
            progress_reward = x_progress * self.forward_weight

            # Add extra bonus if we're making progress toward the absolute max distance
            if x_pos > self.max_x_pos and self.max_x_pos > 2800:
                progress_reward *= 1.5  # 50% bonus when making new progress near the end

            modified_reward += progress_reward

            # Reset counters when moving forward
            self.idle_counter = 0
            self.backward_counter = 0
            self.wall_counter = 0

            # Update last progress position and reset timeout counter
            self.last_progress_x_pos = x_pos
            self.steps_without_progress = 0
        elif x_progress == 0:
            # Penalize standing still
            self.idle_counter += 1

            # Check for potential "stuck at wall" behavior by analyzing action history
            if len(self.last_actions) == self.max_action_history:
                # Check if most actions are the same
                action_counts = {}
                for a in self.last_actions:
                    if a not in action_counts:
                        action_counts[a] = 0
                    action_counts[a] += 1

                # If any action is repeated above threshold, it might be stuck at a wall
                if any(count >= self.same_direction_threshold for count in action_counts.values()):
                    self.wall_counter += 1
                    if self.wall_counter > 5:  # Additional penalty if repeatedly stuck
                        modified_reward -= 3.0

            # Progressive idle penalty that gets worse the longer idle
            if self.idle_counter > 5:
                # More progressive scaling
                idle_factor = min(self.idle_counter / 5,
                                  self.idle_penalty_cap) ** 1.8
                modified_reward -= self.idle_penalty * idle_factor * 3.5

            modified_reward -= self.fixed_idle_penalty
            self.steps_without_progress += 1
        else:
            # Harshly penalize going backward
            self.backward_counter += 1
            backward_penalty = abs(x_progress) * \
                self.backward_penalty_multiplier
            modified_reward -= backward_penalty
            modified_reward -= self.fixed_backward_penalty
            self.steps_without_progress += 1

            # Extra penalty for repeatedly going backward
            if self.backward_counter > 5:
                modified_reward -= self.backward_counter * 0.5  # Progressive penalty

        # 2. Time penalties: more severe as agent approaches max distance
        if x_pos > self.time_optimization_threshold:
            # When near max distance, optimize for time more aggressively
            time_factor = min(1.0, (x_pos - self.time_optimization_threshold) /
                              (self.max_game_distance - self.time_optimization_threshold))
            modified_reward -= self.time_penalty * \
                (1 + time_factor * 5)  # Up to 6x time penalty

            # Mark as having reached near max distance
            if not self.near_max_distance_reached and x_pos > 3000:
                self.near_max_distance_reached = True
                print("Near max distance reached. Optimizing for completion time!")
        else:
            # Standard time penalty otherwise
            modified_reward -= self.time_penalty

        # 3. New max position bonus: MASSIVE reward for reaching new furthest point
        if x_pos > self.max_x_pos:
            # Highly reward discovering new territory
            new_territory_gain = (x_pos - self.max_x_pos)
            territory_bonus = new_territory_gain * self.new_territory_bonus

            # Scale up bonus as we approach the end
            if self.max_x_pos > 2500:
                territory_bonus *= 1.5  # 50% bonus near the end

            modified_reward += territory_bonus
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
            modified_reward -= 150  # Increased penalty
            info['timeout'] = True

        # Also terminate if agent keeps going backward
        if self.backward_counter >= self.max_backward_steps:
            print(
                f"Terminating episode: Going backward for {self.backward_counter} steps")
            done = True
            modified_reward -= 200  # Severe penalty
            info['backward_timeout'] = True

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
        info['backward_counter'] = self.backward_counter
        info['wall_behavior'] = self.wall_counter > 0

        return obs, modified_reward, done, info


# Keep the original class name for compatibility
MarioProgressRewardEnv = customRewardWrapper
