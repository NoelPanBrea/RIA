from p2_seguir_env import RoboboSeguirEnv

env = RoboboSeguirEnv()
import numpy as np
obs, _ = env.reset()
for i in range(6):
    obs, r, t, tr, info = env.step(np.array([-1.0, 1.0]))   # parado, giro máximo a la izquierda
    print(i, 'x =', round(info['x'], 3))