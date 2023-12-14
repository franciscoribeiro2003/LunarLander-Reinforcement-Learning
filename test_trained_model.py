import gymnasium as gym
from stable_baselines3 import PPO


env = gym.make('LunarLander-v2', render_mode="human")
env.reset()

models_dir = "models/PPO_with_custom_rewards_3_10M"
model_path = f"{models_dir}/1100000"

model = PPO.load(model_path, env=env)

episodes = 10

for ep in range(episodes):
    obs, _ = env.reset()
    done = False
    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, truncated, prob = env.step(action.item())
        print(reward)

env.close()
