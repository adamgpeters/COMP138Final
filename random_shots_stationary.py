from rlgym.utils.state_setters import StateSetter
from rlgym.utils.state_setters import StateWrapper
from rlgym.utils.common_values import BLUE_TEAM, ORANGE_TEAM, CEILING_Z, SIDE_WALL_X, BACK_WALL_Y
import numpy as np


class StationaryShots(StateSetter):
    def reset(self, state_wrapper: StateWrapper):
        # Set up our desired spawn location and orientation. Here, we will only change the yaw, leaving the remaining orientation values unchanged.
        x_width = (SIDE_WALL_X - 1152) * 2
        y_width = (BACK_WALL_Y - 1152) * 2
        desired_car_pos = [
            (np.random.rand() - 0.5) * x_width,
            (np.random.rand() - 0.5) * y_width,
            17
        ]  # x, y, z
        desired_ball_pos = [
            (np.random.rand() - 0.5) * x_width,
            (np.random.rand() - 0.5) * y_width,
            17
        ]  # x, y, z

        desired_yaw = np.random.rand() * np.pi * 2

        # # Loop over every car in the game.
        for car in state_wrapper.cars:
            car.set_pos(*desired_car_pos)
            car.set_rot(yaw=desired_yaw)
            car.boost = 1

        state_wrapper.ball.set_pos(*desired_ball_pos)
