""" We attempted to create a conditional reward function that prioritized
    well-placed shots (further from the center of the goal), but we were
    unable to do so properly.
"""

import numpy as np
from rlgym.utils import RewardFunction
from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils import RewardFunction
from rlgym.utils.gamestates import PlayerData, GameState
from rlgym.utils.reward_functions.common_rewards import ConditionalRewardFunction


class BallHeightReward(RewardFunction):
    def __init__(self, own_goal=False):
        super().__init__()
        self.own_goal = own_goal

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # 626.867011415 max distance (about) to corner
        rewardCenter = [0, 0]
        print(state.ball.position)
        twoDBallPos = [state.ball.position[0], state.ball.position[2]]
        reward = np.linalg.norm(
            np.array(rewardCenter) - np.array(twoDBallPos)) / 6
        print(reward)
        return reward


class RewardIfScore(ConditionalRewardFunction):
    def __init__(self, reward_func: RewardFunction, frame_skip=8):
        super().__init__(reward_func)
        self.frame_skip = frame_skip
        self.last_registered_scores = [0, 0]

    def reset(self, initial_state: GameState):
        self.last_registered_values = [0, 0]

    def condition(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> bool:
        if (self.last_registered_scores[0] != state.blue_score or
                self.last_registered_scores[1] != state.orange_score):
            self.last_registered_scores = [
                state.blue_score, state.orange_score]
            return True
        return False
