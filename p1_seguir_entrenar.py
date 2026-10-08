#!/usr/bin/env python3
"""
Practica 1 - Entrenar y probar el seguimiento del guia.

    python3 p1_seguir_entrenar.py --calibrar
    python3 p1_seguir_entrenar.py --comprobar
    python3 p1_seguir_entrenar.py --aleatorio --episodios 5
    python3 p1_seguir_entrenar.py --entrenar --algoritmo sac --pasos 10000
    python3 p1_seguir_entrenar.py --evaluar --episodios 5
"""

import argparse
import os
import time

import numpy as np

from p1_seguir_env import RoboboSeguirEnv, calibrar


RUTA_MODELO = 'p1_seguir_modelo'
RUTA_LOGS = 'p1_seguir_logs'


# =====================================================================
# Hiperparámetros
# =====================================================================
#
# Los valores por defecto de Stable-Baselines3 están pensados para
# entornos simulados que se ejecutan a miles de pasos por segundo. Aquí
# cada paso cuesta tiempo real, de modo que hay que ajustar sobre todo
# el tamaño de los lotes de recogida de datos.

HIPER_SAC = dict(
    learning_rate=3e-4,
    buffer_size=50_000,
    # Pasos que se recogen con acciones aleatorias antes de empezar a
    # entrenar. Llenan el búfer de repetición con algo de variedad.
    learning_starts=500,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    # Una actualización de la red por cada paso del entorno. Es asumible
    # porque el paso del entorno es mucho más lento que la actualización.
    train_freq=1,
    gradient_steps=1,
)

HIPER_PPO = dict(
    learning_rate=3e-4,
    # PPO recoge n_steps transiciones antes de cada actualización. El
    # valor por defecto (2048) supondría diez minutos de simulación entre
    # actualización y actualización.
    n_steps=512,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
)


def construir_algoritmo(nombre, env):
    from stable_baselines3 import PPO, SAC

    if nombre == 'sac':
        return SAC('MlpPolicy', env, verbose=1,
                   tensorboard_log=RUTA_LOGS, **HIPER_SAC)
    if nombre == 'ppo':
        return PPO('MlpPolicy', env, verbose=1,
                   tensorboard_log=RUTA_LOGS, **HIPER_PPO)
    raise ValueError('Algoritmo desconocido: ' + nombre)


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
    try:
        from stable_baselines3.common.monitor import Monitor

        # el Monitor guarda un csv por episodio para las curvas
        os.makedirs(RUTA_LOGS, exist_ok=True)
        registro = os.path.join(
            RUTA_LOGS, '{}_{}'.format(args.algoritmo,
                                      time.strftime('%Y%m%d_%H%M%S')))
        env = Monitor(env, registro, info_keywords=('choque', 'perdido'))

        modelo = construir_algoritmo(args.algoritmo, env)

        print('\nEntrenando {} durante {} pasos.'
              .format(args.algoritmo.upper(), args.pasos))
        print('Se puede interrumpir con Ctrl-C: el modelo se guarda igual.\n')

        t0 = time.time()
        try:
            modelo.learn(total_timesteps=args.pasos, progress_bar=False)
        except KeyboardInterrupt:
            print('\nEntrenamiento interrumpido por el usuario.')

        ruta = RUTA_MODELO + '_' + args.algoritmo
        modelo.save(ruta)
        print('\nModelo guardado en {}.zip'.format(ruta))
        print('Registro en {}.monitor.csv'.format(registro))
        print('Tiempo empleado: {:.1f} min'.format((time.time() - t0) / 60.0))
    finally:
        env.close()


def modo_evaluar(args):
    env = crear_env(args)
    try:
        ruta = RUTA_MODELO + '_' + args.algoritmo
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
    p.add_argument('--pasos-max', type=int, default=200,
                   help='longitud máxima de un episodio')
    p.add_argument('--sin-simulador', action='store_true',
                   help='no usar el modulo sim; el robot se repone a mano')
    p.add_argument('--detalle', action='store_true',
                   help='imprime una línea por paso')

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
