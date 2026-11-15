#!/usr/bin/env python3
"""
Practica 1 - Entrenar y probar el seguimiento del guia con NEAT.

    python3 p2_seguir_entrenar.py --calibrar
    python3 p2_seguir_entrenar.py --comprobar
    python3 p2_seguir_entrenar.py --aleatorio --episodios 5
    python3 p2_seguir_entrenar.py --entrenar --generaciones 20
    python3 p2_seguir_entrenar.py --evaluar --episodios 5
    python3 p2_seguir_entrenar.py --evaluar --modelo mejor_genoma.pkl

Ajuste fino de un genoma ya entrenado (por ejemplo en pygame):
    python3 p2_seguir_entrenar.py --entrenar --desde mejor_genoma.pkl \
        --modelo mejor_genoma_ft.pkl --poblacion 8 --generaciones 4 \
        --episodios-fit 2

--evaluar carga un genoma NEAT guardado en un .pkl, tanto el generado por
el simulador de pygame (sim_seguir_neat.py) como el de --entrenar.
IMPORTANTE: el config.txt debe ser el mismo con el que se entreno
(7 entradas, 2 salidas, misma activacion).
"""

import argparse
import copy
import itertools
import os
import pickle
import time
import math

import neat
import numpy as np

from p2_seguir_env import RoboboSeguirEnv, calibrar
from p2_modelwrapper import RegistroPasos

RUTA_MODELO = 'mejor_genoma2.pkl'
RUTA_LOGS = 'p2_seguir_logs_test'
RUTA_CONFIG = 'config.txt'


# =====================================================================
# NEAT: configuracion, carga de genomas y politica
# =====================================================================

def cargar_config(ruta):
    if not os.path.exists(ruta):
        raise SystemExit('No existe el fichero de configuracion {}.'
                         .format(ruta))
    return neat.config.Config(neat.DefaultGenome,
                              neat.DefaultReproduction,
                              neat.DefaultSpeciesSet,
                              neat.DefaultStagnation,
                              ruta)


def construir_poblacion(config):
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    return population


def poblacion_desde_genoma(config, semilla):
    """Poblacion inicial para el ajuste fino: una copia exacta del genoma
    de partida y el resto copias con una mutacion."""
    population = construir_poblacion(config)
    gc = config.genome_config
    # los nodos nuevos no deben repetir claves de los nodos de la semilla
    gc.node_indexer = itertools.count(max(list(semilla.nodes) + [0]) + 1)
    for i, clave in enumerate(list(population.population)):
        g = copy.deepcopy(semilla)
        g.key = clave
        g.fitness = None
        if i > 0:
            g.mutate(gc)
        population.population[clave] = g
    population.species.speciate(config, population.population,
                                population.generation)
    return population


def jugar_episodio(env, politica):
    """Juega un episodio y devuelve la recompensa total."""
    obs, _ = env.reset()
    total = 0.0
    terminado = truncado = False
    while not (terminado or truncado):
        obs, r, terminado, truncado, _ = env.step(politica(obs))
        total += r
    return total


def guardar_genoma(genoma, ruta):
    carpeta = os.path.dirname(ruta)
    if carpeta:
        os.makedirs(carpeta, exist_ok=True)
    with open(ruta, 'wb') as f:
        pickle.dump(genoma, f)


def cargar_genoma(ruta):
    """Carga un genoma guardado con pickle (simulador o entrenamiento)."""
    if not os.path.exists(ruta):
        raise SystemExit('No existe {}. Hay que entrenar primero o indicar '
                         'el modelo con --modelo.'.format(ruta))
    with open(ruta, 'rb') as f:
        genoma = pickle.load(f)
    if not hasattr(genoma, 'nodes') or not hasattr(genoma, 'connections'):
        raise SystemExit('{} no contiene un genoma de NEAT.'.format(ruta))
    return genoma


def crear_politica(genoma, config):
    """Devuelve una funcion obs -> accion a partir de un genoma."""
    n_in = config.genome_config.num_inputs
    n_out = config.genome_config.num_outputs
    if n_in != 7 or n_out != 2:
        raise SystemExit('El config.txt tiene {} entradas y {} salidas; el '
                         'entorno necesita 7 y 2.'.format(n_in, n_out))
    net = neat.nn.FeedForwardNetwork.create(genoma, config)

    def politica(obs):
        # tanh da valores en [-1, 1]; el entorno ademas hace clip
        return np.asarray(net.activate(list(obs)), dtype=np.float32)

    return politica


# =====================================================================
# Modos de ejecucion
# =====================================================================

def modo_comprobar(args):
    """Verifica que el entorno cumple el contrato de Gymnasium."""
    from stable_baselines3.common.env_checker import check_env

    env = crear_env(args)
    try:
        check_env(env, warn=True)
        print('\nEl entorno cumple la interfaz de Gymnasium.')
    finally:
        env.close()


def modo_aleatorio(args):
    """Ejecuta episodios con acciones aleatorias (linea base)."""
    env = crear_env(args)
    try:
        resumen(env, politica=None, episodios=args.episodios,
                titulo='POLITICA ALEATORIA')
    finally:
        env.close()


def modo_entrenar(args):
    """Entrena con NEAT directamente sobre el entorno (simulador Robobo
    o robot real). Cada genoma juega --episodios-fit episodios.

    Con --desde es un ajuste fino: se parte de un genoma ya entrenado y al
    final se compara con el y se guarda el mejor de los dos."""
    config = cargar_config(args.config)
    if args.poblacion:
        config.pop_size = args.poblacion
    semilla = None
    if args.desde:
        if os.path.abspath(args.desde) == os.path.abspath(args.modelo):
            raise SystemExit('--desde y --modelo no pueden ser el mismo '
                             'fichero.')
        semilla = cargar_genoma(args.desde)

    os.makedirs(RUTA_LOGS, exist_ok=True)
    env = crear_env(args)
    env = RegistroPasos(env, os.path.join(RUTA_LOGS, 'pasos.csv'))
    estado = {'mejor_fit': -1e18}

    def evaluar_genomas(genomes, cfg):
        for _, g in genomes:
            politica = crear_politica(g, cfg)
            # media de varios episodios para que no gane el que tiene suerte
            total = float(np.mean([jugar_episodio(env, politica)
                                   for _ in range(args.episodios_fit)]))
            g.fitness = total
            if total > estado['mejor_fit']:
                estado['mejor_fit'] = total
                guardar_genoma(g, args.modelo)   # guarda el mejor hasta ahora

    try:
        if semilla is not None:
            population = poblacion_desde_genoma(config, semilla)
        else:
            population = construir_poblacion(config)
        winner = population.run(evaluar_genomas, args.generaciones)

        if semilla is not None and args.validar > 0:
            # original contra ajustado, alternados, con episodios nuevos
            pol_orig = crear_politica(semilla, config)
            pol_nuevo = crear_politica(winner, config)
            r_orig, r_nuevo = [], []
            for _ in range(args.validar):
                r_orig.append(jugar_episodio(env, pol_orig))
                r_nuevo.append(jugar_episodio(env, pol_nuevo))
            print('\nValidacion ({} episodios cada uno): original {:.2f}   '
                  'ajustado {:.2f}'.format(args.validar, np.mean(r_orig),
                                           np.mean(r_nuevo)))
            if np.mean(r_nuevo) <= np.mean(r_orig):
                print('El ajuste fino no mejora: se guarda el original.')
                winner = semilla

        guardar_genoma(winner, args.modelo)
        print('\nGenoma guardado en', args.modelo)
    finally:
        env.close()


def modo_evaluar(args):
    """Carga un genoma NEAT (.pkl) y lo ejecuta sin explorar."""
    config = cargar_config(args.config)
    genoma = cargar_genoma(args.modelo)
    politica = crear_politica(genoma, config)

    env = crear_env(args)
    try:
        resumen(env, politica=politica, episodios=args.episodios,
                titulo='GENOMA NEAT ({})'.format(os.path.basename(args.modelo)))
    finally:
        env.close()


# =====================================================================
# Ejecucion de episodios y metricas
# =====================================================================

def resumen(env, politica, episodios, titulo):
    """Ejecuta episodios e imprime como ha ido cada uno."""
    recompensas = []
    longitudes = []
    frac_visible = []
    frac_banda = []
    frac_grad = []
    grad = []
    sep = []
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
        sep_cnt = 0
        n_banda = 0
        n_grad = 0
        grad_cnt = 0
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
            angle = abs(math.degrees(math.atan(info['x'] * math.tan(math.radians(60) / 2))))
            total += r
            pasos += 1
            n_grad += 1 if angle < 15 else 0
            grad_cnt += angle
            n_visible += int(info['visible'])
            n_banda += int(info['en_banda'])
            sep_cnt += info['tam'] 

        t_total += time.time() - t0
        recompensas.append(total)
        longitudes.append(pasos)
        frac_visible.append(100.0 * n_visible / pasos)
        frac_banda.append(100.0 * n_banda / pasos)
        frac_grad.append(100 * n_grad / pasos)
        grad.append(grad_cnt / pasos)
        sep.append(sep_cnt / pasos)


        if info.get('choque'):
            choques += 1
            desenlace = 'choque'
        elif info.get('perdido'):
            perdidos += 1
            desenlace = 'guia perdido'
        else:
            desenlace = 'tiempo agotado'

        print('episodio {:2d}   pasos {:3d}   recompensa {:8.2f}   '
              've la pelota {:5.1f} %   en banda {:5.1f} % separación media {:8.2f} '
              'en ángulo {:5.1f} % ángulo medio {:8.2f}    {}'
              .format(ep + 1, pasos, total, frac_visible[-1],
                      frac_banda[-1], sep[-1], frac_grad[-1], grad[-1], desenlace))

    completos = episodios - choques - perdidos
    print('\nrecompensa media   {:.2f}  (desviacion {:.2f})'
          .format(float(np.mean(recompensas)), float(np.std(recompensas))))
    print('longitud media     {:.1f} pasos'.format(float(np.mean(longitudes))))
    print('duracion del paso  {:.2f} s'.format(t_total / sum(longitudes)))
    print('ve la pelota       {:.1f} % de los pasos'
          .format(float(np.mean(frac_visible))))
    print('en banda           {:.1f} % de los pasos'
          .format(float(np.mean(frac_banda))))
    print('separación media           {:.1f}'
          .format(float(np.mean(sep))))
    print('en angulo           {:.1f} % de los pasos'
          .format(float(np.mean(frac_grad))))
    print('ángulo medio           {:.1f}'
          .format(float(np.mean(grad))))
    print('sin fallo          {}/{}'.format(completos, episodios))
    print('choques            {}/{}'.format(choques, episodios))
    print('guia perdido       {}/{}'.format(perdidos, episodios))


# =====================================================================
# Construccion del entorno y linea de ordenes
# =====================================================================

def crear_env(args):
    return RoboboSeguirEnv(
        pasos_max=args.pasos_max,
        usar_simulador=not args.sin_simulador,
        verbose=args.detalle)


def main():
    p = argparse.ArgumentParser(
        description='Practica 2 de RIA: seguimiento del robot guia (NEAT)')

    modo = p.add_mutually_exclusive_group(required=True)
    modo.add_argument('--calibrar', action='store_true',
                      help='imprime las lecturas del blob y los infrarrojos')
    modo.add_argument('--comprobar', action='store_true',
                      help='verifica la interfaz del entorno')
    modo.add_argument('--aleatorio', action='store_true',
                      help='ejecuta episodios con acciones aleatorias')
    modo.add_argument('--entrenar', action='store_true',
                      help='entrena con NEAT sobre el entorno')
    modo.add_argument('--evaluar', action='store_true',
                      help='carga un genoma .pkl y lo ejecuta')

    p.add_argument('--generaciones', type=int, default=20,
                   help='generaciones de NEAT al entrenar')
    p.add_argument('--desde', default=None,
                   help='genoma .pkl del que partir al entrenar (ajuste fino)')
    p.add_argument('--poblacion', type=int, default=None,
                   help='tamano de la poblacion al entrenar (por defecto, '
                        'el pop_size del config.txt)')
    p.add_argument('--episodios-fit', type=int, default=1,
                   help='episodios por genoma al entrenar (se promedia)')
    p.add_argument('--validar', type=int, default=3,
                   help='con --desde: episodios para comparar al final el '
                        'genoma ajustado con el original (0 = no comparar)')
    p.add_argument('--episodios', type=int, default=5,
                   help='episodios al evaluar / aleatorio')
    p.add_argument('--pasos-max', type=int, default=400,
                   help='longitud maxima de un episodio')
    p.add_argument('--sin-simulador', action='store_true',
                   help='no usar el modulo sim; el robot se repone a mano')
    p.add_argument('--detalle', action='store_true',
                   help='imprime una linea por paso')
    p.add_argument('--modelo', default=RUTA_MODELO,
                   help='genoma .pkl (donde guardar / de donde cargar)')
    p.add_argument('--config', default=RUTA_CONFIG,
                   help='config.txt de NEAT')

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