import sys, os
sys.path.append('/content/UmaMusumeRL')
from game.game_gym_sample_continue import SampleContinueGame
env = SampleContinueGame()
env.reset()

from torch import nn
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy

'''
class CustomPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[256, 512, 256],
                                                          vf=[128, 256, 128])]
                                           )
'''
drive_save_path = "/content/"
save_folder = 'SAC'
model_path = os.path.join(drive_save_path, save_folder)


model = SAC("MlpPolicy", env, verbose=1,
            learning_rate=7e-4, batch_size=512,
            tau=0.01, ent_coef=5, gamma=0.95,
            #   train_freq=100, gradient_steps=100,
            tensorboard_log=model_path)
model.learn(total_timesteps=2000000, log_interval=10)
model.save("sac_pendulum")
model = SAC.load("sac_pendulum")
obs, _ = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info, _ = env.step(action)
