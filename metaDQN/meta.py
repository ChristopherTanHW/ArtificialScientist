import gym
from sbwriter import metaEnv
from stable_baselines3 import DQN
# import wandb
# from wandb.integration.sb3 import WandbCallback
import itertools

print('creating environment...')
env = metaEnv(6, 6, 1, 8, 20, 80, 40)

# run = wandb.init(
#     project="thesis",
#     name='debug: switcing tasks',
#     config=config,
#     sync_tensorboard=True,
#     save_code=True,  # optional
# )

model = DQN("MlpPolicy", metaEnv, verbose=2, buffer_size=10000, 
            learning_starts=1000)

model.learn(total_timesteps=10)


model.save('dqn')