from gymnasium import Wrapper

class CustomLunarLander(Wrapper):
    def __init__(self, env):
        super(CustomLunarLander, self).__init__(env)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        throttle_threshold = 0.5
        left_engine_threshold = 0.5
        right_engine_threshold = 0.5

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
                reward -= 100

        else:
            if obs[0] > -0.3 and obs[0] < 0.3:
                reward -= 0.1
            else:
                reward -= 0.5

             # Shape reward based on actions (throttle and left-right engines)
            main_engine_action = action[0]
            left_right_action = action[1]

            if main_engine_action >= throttle_threshold:
                reward -= 0.2  # Penalize for higher throttle usage

            if abs(left_right_action) >= left_engine_threshold or abs(left_right_action) >= right_engine_threshold:
                reward -= 0.1  # Penalize for excessive left-right engine usage


        return obs, reward, done, truncated, info




# Action is two floats [main engine, left-right engines].
            # Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
            # Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off
        #     self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)
        # else:
        #     # Nop, fire left engine, main engine, right engine
        #     self.action_space = spaces.Discrete(4)
