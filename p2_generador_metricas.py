#!/usr/bin/env python3
"""
Curvas de aprendizaje a partir de los csv del entrenamiento.

Acepta dos tipos de csv (se detecta solo):
  * <registro>.monitor.csv  -> lo escribe Monitor (columnas r, l, t, ...)
  * <registro>.pasos.csv    -> el log por paso (columnas episodio, paso,
                               recompensa, terminado, truncado, ...)

Cada --grupo es una configuracion (nombre + uno o varios csv, uno por
semilla). Todas las curvas se dibujan con los pasos del entorno en el eje
horizontal.

Ejemplos:
    # configuracion final con 3 semillas, banda = desviacion tipica
    python3 p1_curvas.py --grupo "SAC final" s1.monitor.csv s2.monitor.csv \
        s3.monitor.csv --referencia -35.2 --ventana 20

    # comparar con una variante (1 semilla), banda = minimo y maximo
    python3 p1_curvas.py --grupo "SAC final" s1.csv s2.csv s3.csv \
        --grupo "PPO" ppo.csv --banda minmax --salida figuras/curvas
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =====================================================================
# Carga de datos
# =====================================================================

def a_bool(serie):
    return serie.astype(str).str.strip().str.lower().isin(
        ['1', '1.0', 'true'])


def cargar(ruta, causas):
    """Devuelve un DataFrame con una fila por episodio.

    Columnas: r (recompensa), l (pasos), paso (pasos acumulados al
    terminar el episodio), una columna booleana por causa y 'limite'
    (episodio que termina por limite de tiempo: ninguna otra causa).
    """
    df = pd.read_csv(ruta, comment='#')

    if {'r', 'l'} <= set(df.columns):                      # Monitor
        ep = pd.DataFrame({'r': df['r'].astype(float),
                           'l': df['l'].astype(int)})
        for c in causas:
            ep[c] = a_bool(df[c]) if c in df.columns else False

    elif {'episodio', 'recompensa'} <= set(df.columns):    # log por paso
        g = df.groupby('episodio', sort=True)
        completo = (g['terminado'].max() > 0) | (g['truncado'].max() > 0)
        ep = pd.DataFrame({'r': g['recompensa'].sum(),
                           'l': g['recompensa'].count()})
        for c in causas:
            ep[c] = (g[c].max() > 0) if c in df.columns else False
        ep = ep[completo]       # descarta el episodio cortado por Ctrl-C
    else:
        raise SystemExit('{}: formato de csv no reconocido'.format(ruta))

    ep = ep.reset_index(drop=True)
    ep['paso'] = ep['l'].cumsum()
    ep['limite'] = ~ep[causas].any(axis=1) if causas else True
    return ep


# =====================================================================
# Suavizado y agregacion entre semillas
# =====================================================================

def suavizar(serie, ventana):
    """Media movil sobre los ultimos `ventana` episodios."""
    return pd.Series(serie).astype(float).rolling(
        ventana, min_periods=1).mean().to_numpy()


def agregar(xs, ys, puntos=300):
    """Interpola las semillas en una rejilla comun de pasos.

    Devuelve rejilla, media, desviacion tipica, minimo y maximo. La
    rejilla abarca solo el tramo en el que todas las semillas tienen
    datos.
    """
    x0 = max(x[0] for x in xs)
    x1 = min(x[-1] for x in xs)
    rejilla = np.linspace(x0, x1, puntos)
    Y = np.array([np.interp(rejilla, x, y) for x, y in zip(xs, ys)])
    return rejilla, Y.mean(0), Y.std(0), Y.min(0), Y.max(0)


# =====================================================================
# Metricas numericas
# =====================================================================

def metricas_semilla(ep, ventana, causas):
    r_suav = suavizar(ep['r'], ventana)
    i_max = int(np.argmax(r_suav))
    ultimos = ep.tail(ventana)

    # pasos hasta alcanzar el 90 % de la mejora final (suavizada)
    base, final = r_suav[0], r_suav[-1]
    umbral = base + 0.9 * (final - base)
    if final > base:
        i90 = int(np.argmax(r_suav >= umbral))
        pasos_90 = int(ep['paso'].iloc[i90])
    else:
        pasos_90 = np.nan

    m = {
        'episodios': len(ep),
        'pasos_totales': int(ep['paso'].iloc[-1]),
        'rec_final': float(ultimos['r'].mean()),
        'rec_final_std': float(ultimos['r'].std(ddof=0)),
        'rec_max_suav': float(r_suav[i_max]),
        'paso_rec_max': int(ep['paso'].iloc[i_max]),
        'pasos_90pct_mejora': pasos_90,
        'long_final': float(ultimos['l'].mean()),
        'pct_limite_final': 100.0 * float(ultimos['limite'].mean()),
    }
    for c in causas:
        m['pct_{}_final'.format(c)] = 100.0 * float(ultimos[c].mean())
    return m


def imprimir_resumen(nombre, metricas):
    df = pd.DataFrame(metricas)
    print('\n' + nombre + '  ({} semilla{})'.format(
        len(df), 's' if len(df) != 1 else ''))
    print('-' * (len(nombre) + 14))
    for col in df.columns:
        v = df[col].to_numpy(dtype=float)
        if len(v) > 1:
            print('{:22s} {:12.2f} +- {:.2f}'.format(
                col, np.nanmean(v), np.nanstd(v)))
        else:
            print('{:22s} {:12.2f}'.format(col, v[0]))
    return df


# =====================================================================
# Figuras
# =====================================================================

def dibujar(ax, datos, color, etiqueta, valor, ventana, banda,
            escala=1.0):
    """Una curva (con una o varias semillas) en un eje.

    datos: lista de DataFrames de episodios (una semilla cada uno).
    valor: funcion ep -> serie por episodio que se representa.
    """
    xs, ys_suav = [], []
    for ep in datos:
        x = ep['paso'].to_numpy()
        y = np.asarray(valor(ep), dtype=float) * escala
        ax.plot(x, y, color=color, alpha=0.12, lw=0.8)  # sin suavizar
        xs.append(x)
        ys_suav.append(suavizar(y, ventana))

    if len(datos) == 1:
        ax.plot(xs[0], ys_suav[0], color=color, lw=2, label=etiqueta)
        return

    g, media, std, mn, mx = agregar(xs, ys_suav)
    ax.plot(g, media, color=color, lw=2, label=etiqueta)
    if banda == 'std':
        ax.fill_between(g, media - std, media + std, color=color,
                        alpha=0.2, lw=0)
    else:
        ax.fill_between(g, mn, mx, color=color, alpha=0.2, lw=0)


def texto_banda(banda, n_semillas):
    if n_semillas < 2:
        return 'una semilla, sin banda'
    return ('media de {} semillas, banda = '.format(n_semillas) +
            ('+-1 desv. tipica' if banda == 'std' else 'minimo-maximo'))


def figura_recompensa_duracion(grupos, args):
    fig, axs = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    colores = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, (nombre, datos) in enumerate(grupos.items()):
        c = colores[i % len(colores)]
        dibujar(axs[0], datos, c, nombre, lambda e: e['r'],
                args.ventana, args.banda)
        dibujar(axs[1], datos, c, nombre, lambda e: e['l'],
                args.ventana, args.banda,
                escala=args.dur_paso if args.dur_paso else 1.0)

    if args.referencia is not None:
        axs[0].axhline(args.referencia, color='k', ls='--', lw=1,
                       label='politica aleatoria ({:.1f})'
                       .format(args.referencia))

    axs[0].set_ylabel('Recompensa acumulada por episodio\n(adimensional)')
    if args.dur_paso:
        axs[1].set_ylabel('Duracion del episodio (s)\n(1 paso = {:g} s)'
                          .format(args.dur_paso))
    else:
        axs[1].set_ylabel('Duracion del episodio (pasos)')
    axs[1].set_xlabel('Pasos del entorno (llamadas a step())')

    n_max = max(len(d) for d in grupos.values())
    axs[0].set_title('Curva de aprendizaje. Media movil de {} episodios '
                     '(trazo tenue: sin suavizar)\n{}'
                     .format(args.ventana, texto_banda(args.banda, n_max)),
                     fontsize=10)
    for ax in axs:
        ax.grid(alpha=0.3)
    axs[0].legend(loc='best', fontsize=9)
    fig.tight_layout()
    return fig


def figura_causas(grupos, causas, args):
    etiquetas = {'limite': 'limite de tiempo'}
    series = [(c, c) for c in causas] + [('limite', 'limite')]
    n = len(grupos)
    fig, axs = plt.subplots(n, 1, figsize=(9, 3.2 * n + 0.8),
                            sharex=True, sharey=True, squeeze=False)
    colores = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for fila, (nombre, datos) in enumerate(grupos.items()):
        ax = axs[fila, 0]
        for k, (col, _) in enumerate(series):
            dibujar(ax, datos, colores[k % len(colores)],
                    etiquetas.get(col, col),
                    lambda e, col=col: e[col].astype(float),
                    args.ventana, args.banda, escala=100.0)
        ax.set_ylabel('Episodios (%)')
        ax.set_title('{}  ({})'.format(nombre,
                                       texto_banda(args.banda, len(datos))),
                     fontsize=10)
        ax.set_ylim(-3, 103)
        ax.grid(alpha=0.3)
        ax.legend(loc='best', fontsize=8)
    axs[-1, 0].set_xlabel('Pasos del entorno (llamadas a step())')
    fig.suptitle('Causa de fin de episodio (media movil de {} episodios)'
                 .format(args.ventana), fontsize=11)
    fig.tight_layout()
    return fig


# =====================================================================
# Linea de ordenes
# =====================================================================

def main():
    p = argparse.ArgumentParser(
        description='Curvas de aprendizaje de la practica 1')
    p.add_argument('--grupo', nargs='+', action='append', required=True,
                   metavar=('NOMBRE', 'CSV'),
                   help='nombre de la configuracion seguido de uno o '
                        'varios csv (uno por semilla)')
    p.add_argument('--ventana', type=int, default=20,
                   help='tamano de la media movil, en episodios')
    p.add_argument('--banda', choices=['std', 'minmax'], default='std',
                   help='dispersion entre semillas')
    p.add_argument('--referencia', type=float, default=None,
                   help='recompensa media por episodio de la politica '
                        'aleatoria')
    p.add_argument('--dur-paso', type=float, default=None,
                   help='duracion de un paso en segundos; si se indica, '
                        'la duracion se muestra en segundos')
    p.add_argument('--causas', nargs='*', default=['choque', 'perdido'],
                   help='columnas booleanas que indican la causa de fin '
                        '(ademas de limite de tiempo)')
    p.add_argument('--salida', default='curvas',
                   help='prefijo de los ficheros de salida')
    p.add_argument('--mostrar', action='store_true',
                   help='abre las figuras en pantalla')
    args = p.parse_args()

    grupos = {}
    for entrada in args.grupo:
        nombre, rutas = entrada[0], entrada[1:]
        if not rutas:
            raise SystemExit('El grupo "{}" no tiene ningun csv'
                             .format(nombre))
        grupos[nombre] = [cargar(r, args.causas) for r in rutas]

    carpeta = os.path.dirname(args.salida)
    if carpeta:
        os.makedirs(carpeta, exist_ok=True)

    # metricas numericas
    tablas = []
    for nombre, datos in grupos.items():
        m = [metricas_semilla(ep, args.ventana, args.causas)
             for ep in datos]
        df = imprimir_resumen(nombre, m)
        df.insert(0, 'semilla', range(1, len(df) + 1))
        df.insert(0, 'grupo', nombre)
        tablas.append(df)
    pd.concat(tablas).to_csv(args.salida + '_metricas.csv', index=False)
    print('\nMetricas guardadas en {}_metricas.csv'.format(args.salida))

    # figuras
    f1 = figura_recompensa_duracion(grupos, args)
    f2 = figura_causas(grupos, args.causas, args)

    f1.savefig(args.salida + '_recompensa_duracion.png', dpi=200)
    f2.savefig(args.salida + '_causas.png', dpi=200)
    print('Figuras guardadas en {0}_recompensa_duracion.png y '
          '{0}_causas.png'.format(args.salida))

    if args.mostrar:
        plt.show()


if __name__ == '__main__':
    main()