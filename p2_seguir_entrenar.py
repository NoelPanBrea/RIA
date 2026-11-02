#!/usr/bin/env python3
"""
Practica 1 - Entrenar y probar el seguimiento del guia.

    python3 p2_seguir_entrenar.py --calibrar
    python3 p2_seguir_entrenar.py --comprobar
    python3 p2_seguir_entrenar.py --aleatorio --episodios 5
    python3 p2_seguir_entrenar.py --entrenar --algoritmo sac --pasos 10000
    python3 p2_seguir_entrenar.py --evaluar --episodios 5
"""

import argparse
import os
import time
import neat

import numpy as np

from p2_seguir_env import RoboboSeguirEnv, calibrar
from p2_modelwrapper import RegistroPasos

RUTA_MODELO = 'modelo\\p2_seguir_modelo'
RUTA_LOGS = 'p2_seguir_logs_test'
RUTA_CONFIG = "config.txt"

# =====================================================================
# Hiperparámetros
# =====================================================================
#
# Los valores por defecto de Stable-Baselines3 están pensados para
# entornos simulados que se ejecutan a miles de pasos por segundo. Aquí
# cada paso cuesta tiempo real, de modo que hay que ajustar sobre todo
# el tamaño de los lotes de recogida de datos.

def construir_algoritmo():
    config = neat.config.Config(neat.DefaultGenome,
                                neat.DefaultReproduction,
                                neat.DefaultSpeciesSet,
                                neat.DefaultStagnation,
                                RUTA_CONFIG)

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    return population


def cargar_algoritmo(nombre, ruta, env):
    from stable_baselines3 import PPO, SAC

    clase = SAC if nombre == 'sac' else PPO
    return clase.load(ruta, env=env)

# =====================================================================
# Modos de ejecución
# =====================================================================

def modo_comprobar(args):
    """Verifica que el entorno cumple el contrato de Gymnasium.

    check_env recorre los espacios declarados, llama a reset() y a
    step() y avisa de las incoherencias habituales: tipos que no
    coinciden con el espacio, observaciones fuera de los límites
    declarados, valores devueltos en el orden equivocado. Conviene
    ejecutarlo siempre después de tocar el entorno: un fallo aquí se
    manifestaría durante el entrenamiento como una falta de convergencia
    difícil de diagnosticar.
    """
    from stable_baselines3.common.env_checker import check_env

    env = crear_env(args)
    try:
        check_env(env, warn=True)
        print('\nEl entorno cumple la interfaz de Gymnasium.')
    finally:
        env.close()


def modo_aleatorio(args):
    """Ejecuta episodios con acciones aleatorias.

    Establece la línea base. También sirve para comprobar que el
    simulador responde y que el reposicionamiento entre episodios
    funciona.
    """
    env = crear_env(args)
    try:
        resumen(env, politica=None, episodios=args.episodios,
                titulo='POLÍTICA ALEATORIA')
    finally:
        env.close()


def modo_entrenar(args):
    env = crear_env(args)
    env = RegistroPasos(env, "p2_seguir_logs_test/pasos.csv")
    try:
        pop = construir_algoritmo()
        winner = pop.run(evaluar_genomas, args.episodios)
    finally:
        env.close()


def modo_evaluar(args):
    env = crear_env(args)

    def evaluar_genomas(genomes, config):

        nets = []

        for i, g in genomes:
            net = neat.nn.FeedForwardNetwork.create(g, config)
            nets.append(net)
            g.fitness = 0
            terminado, truncado = False, False
            _, _, terminado, truncado, info = env.step
            while not (terminado or truncado):
                accion = net.activate(info[""])
    

    try:
        ruta = args.ruta_modelo + '_' + args.algoritmo
        if not os.path.exists(ruta + '.zip'):
            raise SystemExit(
                'No existe {}.zip. Hay que entrenar primero.'.format(ruta))

        modelo = cargar_algoritmo(args.algoritmo, ruta, env)

        # determinístico=True toma la acción de mayor probabilidad en
        # lugar de muestrear de la distribución. Durante el
        # entrenamiento se muestrea (hace falta explorar); al evaluar,
        # no.
        resumen(env,
                politica=lambda obs: modelo.predict(obs, deterministic=True)[0],
                episodios=args.episodios,
                titulo='POLÍTICA APRENDIDA ({})'.format(args.algoritmo.upper()))
    finally:
        env.close()


# =====================================================================
# Ejecución de episodios y métricas
# =====================================================================

def resumen(env, politica, episodios, titulo):
    """Ejecuta episodios e imprime como ha ido cada uno."""
    recompensas = []
    longitudes = []
    frac_visible = []
    frac_banda = []
    choques = 0
    perdidos = 0
    t_total = 0.0

    print('\n' + titulo)
    print('-' * len(titulo))

    for ep in range(episodios):
        obs, _ = env.reset()
        total = 0.0
        pasos = 0
        n_visible = 0
        n_banda = 0
        info = {}
        terminado = False
        truncado = False
        t0 = time.time()

        while not (terminado or truncado):
            if politica is None:
                accion = env.action_space.sample()
            else:
                accion = politica(obs)
            obs, r, terminado, truncado, info = env.step(accion)
            total += r
            pasos += 1
            n_visible += int(info['visible'])
            n_banda += int(info['en_banda'])

        t_total += time.time() - t0
        recompensas.append(total)
        longitudes.append(pasos)
        frac_visible.append(100.0 * n_visible / pasos)
        frac_banda.append(100.0 * n_banda / pasos)

        if info.get('choque'):
            choques += 1
            desenlace = 'choque'
        elif info.get('perdido'):
            perdidos += 1
            desenlace = 'guia perdido'
        else:
            desenlace = 'tiempo agotado'

        print('episodio {:2d}   pasos {:3d}   recompensa {:8.2f}   '
              've la pelota {:5.1f} %   en banda {:5.1f} %   {}'
              .format(ep + 1, pasos, total, frac_visible[-1],
                      frac_banda[-1], desenlace))

    completos = episodios - choques - perdidos
    print('\nrecompensa media   {:.2f}  (desviacion {:.2f})'
          .format(float(np.mean(recompensas)), float(np.std(recompensas))))
    print('longitud media     {:.1f} pasos'.format(float(np.mean(longitudes))))
    print('duracion del paso  {:.2f} s'.format(t_total / sum(longitudes)))
    print('ve la pelota       {:.1f} % de los pasos'
          .format(float(np.mean(frac_visible))))
    print('en banda           {:.1f} % de los pasos'
          .format(float(np.mean(frac_banda))))
    print('sin fallo          {}/{}'.format(completos, episodios))
    print('choques            {}/{}'.format(choques, episodios))
    print('guia perdido       {}/{}'.format(perdidos, episodios))



# =====================================================================
# Construcción del entorno y línea de órdenes
# =====================================================================

def crear_env(args):
    return RoboboSeguirEnv(
        pasos_max=args.pasos_max,
        usar_simulador=not args.sin_simulador,
        verbose=args.detalle)


def main():
    p = argparse.ArgumentParser(
        description='Practica 1 de RIA: seguimiento del robot guia')

    modo = p.add_mutually_exclusive_group(required=True)
    modo.add_argument('--calibrar', action='store_true',
                      help='imprime las lecturas de los infrarrojos')
    modo.add_argument('--comprobar', action='store_true',
                      help='verifica la interfaz del entorno')
    modo.add_argument('--aleatorio', action='store_true',
                      help='ejecuta episodios con acciones aleatorias')
    modo.add_argument('--entrenar', action='store_true')
    modo.add_argument('--evaluar', action='store_true')

    p.add_argument('--algoritmo', choices=['sac', 'ppo'], default='sac')
    p.add_argument('--pasos', type=int, default=10_000,
                   help='pasos totales de entrenamiento')
    p.add_argument('--episodios', type=int, default=5)
    p.add_argument('--pasos-max', type=int, default=400,
                   help='longitud máxima de un episodio')
    p.add_argument('--sin-simulador', action='store_true',
                   help='no usar el modulo sim; el robot se repone a mano')
    p.add_argument('--detalle', action='store_true',
                   help='imprime una línea por paso')
    p.add_argument('--ruta-modelo', default=RUTA_MODELO)

    args = p.parse_args()

    if args.calibrar:
        calibrar()
    elif args.comprobar:
        modo_comprobar(args)
    elif args.aleatorio:
        modo_aleatorio(args)
    elif args.entrenar:
        modo_entrenar(args)
    elif args.evaluar:
        modo_evaluar(args)


if __name__ == '__main__':
    main()
