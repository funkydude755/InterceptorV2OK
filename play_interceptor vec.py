

from Interceptor_V2gameCNN import InterceptorV2Env
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, ACKTR
from stable_baselines.common.policies import MlpPolicy,MlpLnLstmPolicy, CnnLstmPolicy, CnnPolicy, ActorCriticPolicy

from stable_baselines.logger import configure
from stable_baselines.common.vec_env import VecFrameStack
configure()



num_envs = 2


env = DummyVecEnv([InterceptorV2Env for i in range(num_envs)])

# vec_env = VecFrameStack(env, n_stack=num_envs)
#, full_tensorboard_log=True
# Init()
model = PPO2(CnnPolicy, env, verbose=1, tensorboard_log='/home/ok/OAI/logs/')

#LOAD A TRAINED MODEL
model.load("/home/ok/OAI/interceptor_model_ppo_CNN_10000.pkl", env=env)

#Train A MODEL
model.learn(total_timesteps=10000000, tb_log_name="interceptor_model_PPO2_10000CNN")

#Save A MODEL
model.save("/home/ok/OAI/interceptor_model_ppo_CNN_10000")




# Play the model
obs = env.reset()

# for i in range(1000):
#     action, _states = model.predict(obs)
#     # print (action)
#     obs, rewards, dones, info = env.step(action)
#     print(rewards)
#     env.render()
#

