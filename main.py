import rlgym
from stable_baselines3 import PPO


"""
DEFAULTS IN MAKE

game_speed: int = 100,
tick_skip: int = 8,
spawn_opponents: bool = True,
self_play: bool = False,
random_resets: bool = False,
team_size: int = 1,
terminal_conditions: List[object] = (TimeoutCondition(225), GoalScoredCondition()),
reward_fn: object = DefaultReward(),
obs_builder: object = DefaultObs(),
action_parser: object = DefaultAction(),
state_setter: object = DefaultState(),
launch_preference: str = LaunchPreference.EPIC,
path_to_rl: str = None,
use_injector: bool = False
force_paging: bool = False
"""

#Make the default rlgym environment
env = rlgym.make()

#Initialize PPO from stable_baselines3"
"""
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
"""
model = PPO("MlpPolicy", env=env, verbose=1)

#Train our agent!
model.learn(total_timesteps=int(1e6))
