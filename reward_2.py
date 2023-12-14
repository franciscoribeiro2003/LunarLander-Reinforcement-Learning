from gymnasium import Wrapper

class CustomLunarLander(Wrapper):
    def __init__(self, env):
        super(CustomLunarLander, self).__init__(env)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        if done:
            if obs[0] > -0.1 and obs[0] < 0.1  and obs[3] > -0.5 and obs[4] < 0.5 and abs(obs[5]) < 0.05:
                reward += 130  # Recompensa por mover da parte superior para a plataforma e parar
                if obs[6] == 1 and obs[7] == 1:
                    reward += 20  # Recompensa por cada perna com contato no solo

            elif obs[0] > -0.3 and obs[0] < 0.3 and obs[3] > -0.5 and obs[4] < 0.5 and abs(obs[5]) < 0.05:
                reward += 100
                if obs[6] == 1 and obs[7] == 1:
                    reward += 20

            else:
                reward -= 130

        return obs, reward, done, truncated, info
