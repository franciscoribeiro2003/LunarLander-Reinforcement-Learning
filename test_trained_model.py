import gymnasium as gym
from stable_baselines3 import PPO
from argparse          import ArgumentParser
from sys               import argv, exit

env = gym.make('LunarLander-v2', render_mode="human")

def main():
    parser = ArgumentParser(description='Please specify which model to test')

    parser.add_argument( '-o', '--original', action='store_true', help='Test the original model')
    parser.add_argument( '-c', '--custom', action='store_true', help='Test the custom model')

    if len(argv) == 1:
        parser.print_help()
        exit(0)

    args = parser.parse_args()

    env.reset()

    if args.original:
        models_dir = "models/PPO_no_modification_10M"
        model_path = f"{models_dir}/6000000"

    elif args.custom:
        models_dir = "models/PPO_with_custom_rewards_1_10M"
        model_path = f"{models_dir}/5430000"

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


if __name__ == "__main__":
    main()
