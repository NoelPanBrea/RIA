import csv
import gymnasium as gym
import numpy as np

class RegistroPasos(gym.Wrapper):
    """Escribe una fila por cada step() en un csv."""

    def __init__(self, env, ruta):
        super().__init__(env)
        self._f = open(ruta, 'w', newline='')
        self._w = csv.writer(self._f)
        n_act = int(np.prod(env.action_space.shape))
        self._w.writerow(
            ['episodio', 'paso', 'recompensa', 'terminado', 'truncado',
             'visible', 'en_banda', 'choque', 'perdido']
            + ['accion_{}'.format(i) for i in range(n_act)])
        self._ep = 0
        self._paso = 0

    def reset(self, **kwargs):
        self._ep += 1
        self._paso = 0
        return self.env.reset(**kwargs)

    def step(self, accion):
        obs, r, term, trunc, info = self.env.step(accion)
        self._paso += 1
        self._w.writerow(
            [self._ep, self._paso, r, int(term), int(trunc),
             int(info.get('visible', 0)), int(info.get('en_banda', 0)),
             int(info.get('choque', 0)), int(info.get('perdido', 0))]
            + list(np.ravel(accion)))
        self._f.flush()  # así no se pierde nada si haces Ctrl-C
        return obs, r, term, trunc, info

    def close(self):
        self._f.close()
        super().close()