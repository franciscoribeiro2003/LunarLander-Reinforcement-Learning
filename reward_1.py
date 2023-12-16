from gymnasium import Wrapper

class CustomLunarLander(Wrapper):
    def __init__(self, env):
        super(CustomLunarLander, self).__init__(env)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        if done:
            if obs[0] > -0.1 and obs[0] < 0.1  and obs[3] > -0.5:
                reward += 10  # Recompensa por mover da parte superior para a plataforma e parar
                if obs[6] == 1 and obs[7] == 1:
                    reward += 1.5  # Recompensa por cada perna com contato no solo

            elif obs[0] > -0.3 and obs[0] < 0.3 and obs[3] > -0.5 and obs[4] < 0.5 and abs(obs[5]) < 0.05:
                reward += 7 # Recompensa por estar na platoforma e parar bem.
                if obs[6] == 1 and obs[7] == 1:
                    reward += 1.5 # Recompensa por cada perna com contato no solo

            else:
                reward -= 10 # Penalidade por cair

            reward += (1 - abs(obs[4])) * 5 # Recompensa por estar centralizado na plataforma

        else:
            reward -= 0.2 # Penalidade por cada passo

        return obs, reward, done, truncated, info
