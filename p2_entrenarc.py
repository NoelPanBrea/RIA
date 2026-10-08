#!/usr/bin/env python3
"""
Práctica 1 — Entrenamiento y evaluación rápida del seguidor.

Usa Stable-Baselines3 sobre el entorno de p1_entorno.py.

USO
  1. Calibrar el sensor (CSV + resumen):      --calibrar
  2. (Opcional) grabar la línea central:      --linea
  3. Comprobar la interfaz de Gymnasium:      --comprobar
  4. Línea base aleatoria:                    --aleatorio --episodios 8
  5. Entrenar:                                --entrenar --algoritmo sac \
                                               --pasos 10000 --semilla 0
  6. Evaluar (determinista, inicio sin aleatorizar):
                                              --evaluar --episodios 8

Ejemplo de presupuesto: con ~0.2 s por paso, 10 000 pasos son ~35 min.
Medir los segundos por paso reales (info['dt']) antes de fijarlo.

NOTA: este script NO implementa el protocolo del anexo B (cuatro
configuraciones del guía x 2 episodios de 60 s con métricas de la verdad
del terreno). --evaluar es una comprobación rápida con los criterios del
propio entorno. El guía se configura en el simulador; declarar la
configuración usada en el entrenamiento en la memoria.
"""

import argparse
import collections
import os
import time

import numpy as np

from p2_robobo_envc import (RoboboSeguidorEnv, calibrar, grabar_linea_central, cargar_config)

RUTA_LOGS = 'p1_logs'
RUTA_MODELOS = 'p1_modelos'


# =====================================================================
# Hiperparámetros
# =====================================================================
# Cada paso cuesta tiempo real (~0.2 s): se ajusta el tamaño de los
# lotes de recogida y se prefiere SAC (fuera de política) por su
# eficiencia en muestras.

HIPER_SAC = dict(
    learning_rate=3e-4,
    buffer_size=50_000,
    learning_starts=500,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
)

HIPER_PPO = dict(
    learning_rate=3e-4,
    n_steps=512,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
)


def nombre_modelo(args):
    return os.path.join(
        RUTA_MODELOS, '{}_s{}'.format(args.algoritmo, args.semilla))


def construir_algoritmo(args, env):
    from stable_baselines3 import PPO, SAC
    if args.algoritmo == 'sac':
        return SAC('MlpPolicy', env, verbose=1, seed=args.semilla,
                   tensorboard_log=RUTA_LOGS, **HIPER_SAC)
    return PPO('MlpPolicy', env, verbose=1, seed=args.semilla,
               tensorboard_log=RUTA_LOGS, **HIPER_PPO)


def cargar_algoritmo(args, env):
    from stable_baselines3 import PPO, SAC
    clase = SAC if args.algoritmo == 'sac' else PPO
    return clase.load(nombre_modelo(args), env=env)


# =====================================================================
# Modos
# =====================================================================

def modo_comprobar(args):
    """Verifica el contrato de Gymnasium (no mueve nada más que el robot
    durante unos pasos aleatorios)."""
    from stable_baselines3.common.env_checker import check_env
    env = crear_env(args)
    try:
        check_env(env, warn=True)
        print('\nEl entorno cumple la interfaz de Gymnasium.')
    finally:
        env.close()


def modo_aleatorio(args):
    env = crear_env(args, evaluacion=True)
    try:
        resumen(env, None, args.episodios, 'POLÍTICA ALEATORIA',
                semilla=args.semilla)
    finally:
        env.close()


def modo_entrenar(args):
    from stable_baselines3.common.monitor import Monitor
    os.makedirs(RUTA_LOGS, exist_ok=True)
    os.makedirs(RUTA_MODELOS, exist_ok=True)

    env = crear_env(args)
    try:
        # Registro por episodio (apartado 9.2): recompensa, longitud,
        # tiempo y causa de fin. Eje horizontal de las curvas: pasos de
        # entorno (acumular la columna 'l').
        csv_ep = os.path.join(
            RUTA_LOGS, 'episodios_{}_s{}'.format(args.algoritmo, args.semilla))
        env = Monitor(env, filename=csv_ep, info_keywords=('causa',))

        modelo = construir_algoritmo(args, env)
        print('\nEntrenando {} durante {} pasos (semilla {}).'
              .format(args.algoritmo.upper(), args.pasos, args.semilla))
        print('Ctrl-C interrumpe y guarda el modelo.\n')

        t0 = time.time()
        try:
            modelo.learn(total_timesteps=args.pasos, progress_bar=False)
        except KeyboardInterrupt:
            print('\nEntrenamiento interrumpido por el usuario.')

        modelo.save(nombre_modelo(args))
        print('\nModelo guardado en {}.zip'.format(nombre_modelo(args)))
        print('Tiempo empleado: {:.1f} min ({:.3f} s/paso)'.format(
            (time.time() - t0) / 60.0,
            (time.time() - t0) / max(modelo.num_timesteps, 1)))
    finally:
        env.close()


def modo_evaluar(args):
    if not os.path.exists(nombre_modelo(args) + '.zip'):
        raise SystemExit('No existe {}.zip. Entrenar primero.'
                         .format(nombre_modelo(args)))
    env = crear_env(args, evaluacion=True)
    try:
        modelo = cargar_algoritmo(args, env)
        resumen(env,
                lambda obs: modelo.predict(obs, deterministic=True)[0],
                args.episodios,
                'POLÍTICA APRENDIDA ({}, semilla {})'.format(
                    args.algoritmo.upper(), args.semilla),
                semilla=args.semilla)
    finally:
        env.close()


# =====================================================================
# Episodios y métricas
# =====================================================================

def resumen(env, politica, episodios, titulo, semilla=0):
    recompensas, longitudes, dts = [], [], []
    causas = collections.Counter()

    print('\n' + titulo)
    print('-' * len(titulo))

    for ep in range(episodios):
        obs, _ = env.reset(seed=semilla + ep)
        total, pasos = 0.0, 0
        info = {}
        fin = False
        while not fin:
            accion = (env.action_space.sample() if politica is None
                      else politica(obs))
            obs, r, terminado, truncado, info = env.step(accion)
            total += r
            pasos += 1
            dts.append(info.get('dt', 0.0))
            fin = terminado or truncado

        causa = info.get('causa') or 'desconocida'
        causas[causa] += 1
        recompensas.append(total)
        longitudes.append(pasos)
        print('episodio {:2d}   pasos {:3d}   recompensa {:8.2f}   {}'
              .format(ep + 1, pasos, total, causa))

    print('\nrecompensa media  {:.2f}  (desviación {:.2f})'.format(
        float(np.mean(recompensas)), float(np.std(recompensas))))
    print('longitud media    {:.1f} pasos'.format(float(np.mean(longitudes))))
    print('periodo de paso   {:.3f} s'.format(float(np.mean(dts))))
    print('completados       {}/{}'.format(causas['limite_tiempo'], episodios))
    for c, n in causas.items():
        if c != 'limite_tiempo':
            print('  {:<16s}{}/{}'.format(c, n, episodios))


# =====================================================================
# Entorno y línea de órdenes
# =====================================================================

def crear_env(args, evaluacion=False):
    cfg = cargar_config(args.config)
    if args.pasos_max is not None:
        cfg['pasos_max'] = args.pasos_max
    return RoboboSeguidorEnv(
        config=cfg,
        usar_simulador=not args.sin_simulador,
        modo_evaluacion=evaluacion,
        verbose=args.detalle)


def main():
    p = argparse.ArgumentParser(
        description='Práctica 1 de RIA: seguimiento de un vehículo guía')

    modo = p.add_mutually_exclusive_group(required=True)
    modo.add_argument('--calibrar', action='store_true')
    modo.add_argument('--linea', action='store_true',
                      help='graba la línea central (mueve al seguidor)')
    modo.add_argument('--comprobar', action='store_true')
    modo.add_argument('--aleatorio', action='store_true')
    modo.add_argument('--entrenar', action='store_true')
    modo.add_argument('--evaluar', action='store_true')

    p.add_argument('--algoritmo', choices=['sac', 'ppo'], default='sac')
    p.add_argument('--pasos', type=int, default=10_000)
    p.add_argument('--episodios', type=int, default=8)
    p.add_argument('--semilla', type=int, default=0)
    p.add_argument('--pasos-max', type=int, default=None,
                   help='sustituye pasos_max del archivo de configuración')
    p.add_argument('--config', default='config_p1.json')
    p.add_argument('--sin-simulador', action='store_true',
                   help='robot real: el reinicio lo hace una persona')
    p.add_argument('--detalle', action='store_true')

    args = p.parse_args()

    if args.calibrar:
        calibrar(args.config)
    elif args.linea:
        grabar_linea_central(args.config)
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