import numpy as np
import math

from rlgym.utils import RewardFunction, math
from rlgym.utils.common_values import BLUE_TEAM, ORANGE_TEAM, ORANGE_GOAL_BACK, \
    BLUE_GOAL_BACK, BALL_MAX_SPEED, BACK_WALL_Y, BALL_RADIUS, BACK_NET_Y
from rlgym.utils.gamestates import GameState, PlayerData

from abc import abstractmethod

from rlgym.utils import RewardFunction
from rlgym.utils.common_values import BLUE_TEAM, ORANGE_TEAM
from rlgym.utils.gamestates import PlayerData, GameState
from rlgym.utils.reward_functions.common_rewards import ConditionalRewardFunction


class BallHeightReward(RewardFunction):
    def __init__(self, own_goal=False):
        super().__init__()
        self.own_goal = own_goal
        # self.prev_ball_pos = [0, 0]

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # 626.867011415 max distance (about) to corner
        rewardCenter = [0, 0]
        print(state.ball.position)
        # print(twoDBallPos)
        twoDBallPos = [state.ball.position[0], state.ball.position[2]]
        reward = np.linalg.norm(
            np.array(rewardCenter) - np.array(twoDBallPos)) / 6
        print(reward)
        return reward


class RewardIfScore(ConditionalRewardFunction):
    def __init__(self, reward_func: RewardFunction, team_only=True):
        super().__init__(reward_func)
        self.team_only = team_only
        self.last_registered_scores = [0, 0]

    def reset(self, initial_state: GameState):
        self.last_registered_values = [0, 0]  # ?

    def condition(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> bool:
        if (self.last_registered_scores[0] != state.blue_score or self.last_registered_scores[1] != state.orange_score):
            self.last_registered_scores = [
                state.blue_score, state.orange_score]
            return True
        return False

        # dist = np.linalg.norm(player.car_data.position - state.ball.position)
        # for player2 in state.players:
        #     if not self.team_only or player2.team_num == player.team_num:
        #         dist2 = np.linalg.norm(
        #             player2.car_data.position - state.ball.position)
        #         if dist2 < dist:
        #             return False
        # return True
