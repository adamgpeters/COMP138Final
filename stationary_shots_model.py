import numpy as np
from rlgym.envs import Match
from rlgym.utils.action_parsers import DiscreteAction
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecCheckNan
from stable_baselines3.ppo import MlpPolicy
from rlgym.utils.obs_builders import AdvancedObs
from random_shots_stationary import StationaryShots
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv
from rlgym.utils.reward_functions.common_rewards.misc_rewards import EventReward
from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import VelocityPlayerToBallReward
from rlgym.utils.reward_functions.common_rewards.ball_goal_rewards import VelocityBallToGoalReward
from rlgym.utils.reward_functions import CombinedReward


if __name__ == '__main__':  # Required for multiprocessing
    frame_skip = 8          # Number of ticks to repeat an action
    # Easier to conceptualize, after this many seconds the reward discount is 0.5
    half_life_seconds = 5

    fps = 120 / frame_skip
    gamma = np.exp(np.log(0.5) / (fps * half_life_seconds))
    agents_per_match = 2
    num_instances = 1
    target_steps = 1_000_000
    # making sure the experience counts line up properly
    steps = target_steps // (num_instances * agents_per_match)
    # getting the batch size down to something more manageable - 100k in this case
    batch_size = target_steps // 10
    training_interval = 25_000_000
    mmr_save_frequency = 50_000_000
    name_prefix = "stationary_small_from_scratch"
    save_dir = "models/" + name_prefix + "/"

    def exit_save(model):
        model.save(save_dir + "exit_save")

    def get_match():  # Need to use a function so that each instance can call it and produce their own objects
        return Match(
            team_size=1,
            tick_skip=frame_skip,
            reward_function=CombinedReward(
                (
                    VelocityPlayerToBallReward(),
                    VelocityBallToGoalReward(),
                    EventReward(
                        team_goal=100.0,
                        concede=-100.0,
                        shot=5.0,
                    ),
                ),
                (0.1, 1.0, 1.0)),
            spawn_opponents=False,
            terminal_conditions=[
                TimeoutCondition(fps * 15),  # 15 seconds to score
                GoalScoredCondition()],
            obs_builder=AdvancedObs(),
            state_setter=StationaryShots(),
            action_parser=DiscreteAction()
        )

    # Start 1 instances, waiting 60 seconds between each
    env = SB3MultipleInstanceEnv(get_match, num_instances)
    env = VecCheckNan(env)
    # Logs mean reward and ep_len to Tensorboard
    env = VecMonitor(env)
    # Normalizes rewards
    env = VecNormalize(env, norm_obs=False, gamma=gamma)

    try:
        model = PPO.load(
            save_dir + "exit_save.zip",
            env,
            device="auto",
            # automatically adjusts to users changing instance count, may encounter shaping error otherwise
            custom_objects={"n_envs": env.num_envs},
        )
        print("Loaded previous exit save.")
    except:
        print("No saved model found, creating new model.")
        from torch.nn import Tanh
        policy_kwargs = dict(
            activation_fn=Tanh,
            net_arch=[64, 64, dict(pi=[32, 32, 32], vf=[32, 32, 32])],
        )

        model = PPO(
            MlpPolicy,
            env,
            n_epochs=10,                 # PPO calls for multiple epochs
            policy_kwargs=policy_kwargs,
            learning_rate=5e-5,          # Around this is fairly common for PPO
            ent_coef=0.01,               # From PPO Atari
            vf_coef=1.,                  # From PPO Atari
            gamma=gamma,                 # Gamma as calculated using half-life
            verbose=3,                   # Print out all the info as we're going
            batch_size=batch_size,       # Batch size as high as possible within reason
            n_steps=steps,               # Number of steps to perform before optimizing network
            # `tensorboard --logdir out/logs` in terminal to see graphs
            tensorboard_log="logs_" + name_prefix,
            device="auto"                # Uses GPU if available
        )

    # Save model every so often
    # Divide by num_envs (number of agents) because callback only increments every time all agents have taken a step
    callback = CheckpointCallback(
        round(5_000_000 / env.num_envs), save_path=save_dir, name_prefix=name_prefix)

    try:
        mmr_model_target_count = model.num_timesteps + mmr_save_frequency
        while True:
            model.learn(training_interval, callback=callback,
                        reset_num_timesteps=False)
            model.save(save_dir + "exit_save")
            if model.num_timesteps >= mmr_model_target_count:
                model.save(f"mmr_models/{model.num_timesteps}")
                mmr_model_target_count += mmr_save_frequency

    except KeyboardInterrupt:
        print("Exiting training")

    print("Saving model")
    exit_save(model)
    print("Save complete")
