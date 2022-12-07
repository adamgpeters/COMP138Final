from rlgym.utils.state_setters import StateSetter
from rlgym.utils.state_setters import StateWrapper
from rlgym.utils.common_values import BACK_WALL_Y
import numpy as np


class DeterministicShot(StateSetter):
    def reset(self, state_wrapper: StateWrapper):
        # Set up our desired spawn location and orientation. Here, we will only change the yaw, leaving the remaining orientation values unchanged.
        desired_car_pos = [
            0,
            0,
            17
        ]  # x, y, z
        desired_ball_pos = [
            0,
            BACK_WALL_Y / 2,
            17
        ]  # x, y, z

        desired_yaw = 0.5 * np.pi

        # # Loop over every car in the game.
        for car in state_wrapper.cars:
            car.set_pos(*desired_car_pos)
            car.set_rot(yaw=desired_yaw)
            car.boost = 1

        state_wrapper.ball.set_pos(*desired_ball_pos)
