from gymnasium import Wrapper

class CustomLunarLander(Wrapper):
    def __init__(self, env):
        super(CustomLunarLander, self).__init__(env)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        if done:
            if obs[0] >= -0.3 and obs[0] <= 0.3:
                reward += 100
                if obs[0] >= -0.1 and obs[0] <= 0.1:
                    reward += 50
                if obs[6] == 1 and obs[7] == 1:
                    reward += 50
                elif obs[6] == 1 or obs[7] == 1:
                    reward -= 15
                elif obs[6] == 0 and obs[7] == 0:
                    reward -= 50

            elif obs[0] >= -0.6 and obs[0] <= 0.6:
                reward -= 100

            else:
                reward -= 150

        else:
            if obs[0] >= -0.3 and obs[0] <= 0.3:
                reward += 5
                if obs[4] >= 0.5 and obs[5] <= -0.5:
                    reward -= 3
            else:
                reward -= 5

        return obs, reward, done, truncated, info
