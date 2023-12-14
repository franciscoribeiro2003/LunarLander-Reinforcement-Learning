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
                reward += 7
                if obs[6] == 1 and obs[7] == 1:
                    reward += 1.5

            else:
                reward -= 10

            reward += (1 - abs(obs[4])) * 5

        else:
            reward -= 0.1

        return obs, reward, done, truncated, info




            # Se o lander se moveu para longe da plataforma ou colidiu
             # Penalidade por sair da plataforma ou colidir

            # if obs[0] >= -0.3 and obs[0] <= 0.3: # coordinates inside the landing site
            #     reward = 5
            #     if obs[0] >= -0.1 and obs[0] <= 0.1: # coordinates in perfect center
            #         reward = 10
            #     if obs[6] == 1 and obs[7] == 1: # both legs touching the ground
            #         reward = 15
            #         if obs[0] >= -0.1 and obs[0] <= 0.1:
            #             reward = 20
            #     elif obs[6] == 1 or obs[7] == 1:  # one leg touching the ground
            #         reward = 3
            #     elif obs[6] == 0 and obs[7] == 0:  # no legs touching the ground
            #         reward = 0

            # elif obs[0] >= -0.6 and obs[0] <= 0.6:  # coordinates to be closer to the landing site
            #     reward = -15

            # else:
            #     reward = -20

        """
        else:
            if obs[0] >= -0.3 and obs[0] <= 0.3: # coordinates inside the landing site
                reward = - 5
                if obs[4] >= 0.5 and obs[5] <= -0.5:
                    reward = - 7 # penalize the lander for rotating
            else:
                reward = - 10
         """
