#!/usr/bin/env python3
"""
Curvas de evolucion de NEAT a partir de los CSV que escribe RegistroGeneracion.

Estructura esperada (una carpeta por semilla):

    p2_seguir_logs_test/
        semilla_1/generaciones.csv
        semilla_1/generaciones_especies.csv
        semilla_1/meta.json            {"poblacion":..,"episodios_fit":..,"pasos_max":..,"semilla":..}
        semilla_2/...
        referencia.json                {"calidad": 123.4}     (opcional)

Uso:
    python3 visualizar_evolucion.py
    python3 visualizar_evolucion.py --carpeta p2_seguir_logs_test --referencia 250
    python3 visualizar_evolucion.py --banda std --q-min -280 --q-max 680

Genera en --salida (por defecto 'figuras'):
    fig1_calidad.(png|pdf)        mejor y media de la poblacion, con bandas
    fig2_especies.(png|pdf)       tamano de las especies (areas apiladas), una fila por semilla
    fig3_complejidad.(png|pdf)    nodos y conexiones del mejor individuo
    resumen.txt                   cuando supera la referencia, estancamiento, especies
"""

import argparse
import csv
import glob
import json
import os
import textwrap

# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# =====================================================================
# Carga de datos
# =====================================================================

def leer_csv(ruta):
    with open(ruta, newline='') as f:
        filas = list(csv.DictReader(f))
    if not filas:
        return {}
    return {k: np.array([float(r[k]) for r in filas]) for k in filas[0]}


def cargar_semillas(carpeta):
    semillas = []
    for d in sorted(glob.glob(os.path.join(carpeta, 'semilla_*'))):
        ruta = os.path.join(d, 'generaciones.csv')
        if not os.path.exists(ruta):
            continue
        datos = leer_csv(ruta)
        if not datos:
            continue

        especies = {}
        ruta_esp = os.path.join(d, 'generaciones_especies.csv')
        if os.path.exists(ruta_esp):
            with open(ruta_esp, newline='') as f:
                for r in csv.DictReader(f):
                    g, s, t = int(r['generacion']), int(r['especie']), int(r['tamano'])
                    especies.setdefault(s, {})[g] = t

        meta = {}
        ruta_meta = os.path.join(d, 'meta.json')
        if os.path.exists(ruta_meta):
            with open(ruta_meta) as f:
                meta = json.load(f)

        semillas.append({
            'nombre': os.path.basename(d),
            'datos': datos,
            'especies': especies,
            'meta': meta,
        })
    if not semillas:
        raise SystemExit('No se han encontrado carpetas semilla_*/generaciones.csv '
                         'en {}.'.format(carpeta))
    return semillas


def cargar_referencia(args):
    if args.referencia is not None:
        return args.referencia
    ruta = os.path.join(args.carpeta, 'referencia.json')
    if os.path.exists(ruta):
        with open(ruta) as f:
            return float(json.load(f)['calidad'])
    return None


# =====================================================================
# Utilidades
# =====================================================================

def apilar(semillas, campo):
    """Matriz (semillas x generaciones), recortada a la ejecucion mas corta."""
    n = min(len(s['datos'][campo]) for s in semillas)
    return np.vstack([s['datos'][campo][:n] for s in semillas])


def agregar(matriz, modo):
    media = matriz.mean(axis=0)
    if modo == 'std':
        d = matriz.std(axis=0)
        return media, media - d, media + d
    return media, matriz.min(axis=0), matriz.max(axis=0)


def texto_banda(modo):
    return ('desviacion tipica entre semillas' if modo == 'std'
            else 'minimo y maximo entre semillas')


def datos_pie(semillas):
    """Texto con tamano de poblacion, episodios, pasos y horas."""
    meta = semillas[0]['meta']
    pasos = sum(s['datos']['pasos'][-1] for s in semillas)
    horas = sum(s['datos']['segundos'][-1] for s in semillas) / 3600.0
    partes = []
    if 'poblacion' in meta:
        partes.append('poblacion de {} individuos'.format(meta['poblacion']))
    if 'episodios_fit' in meta:
        partes.append('{} episodio(s) por individuo'.format(meta['episodios_fit']))
    if 'pasos_max' in meta:
        partes.append('episodios de como maximo {} pasos'.format(meta['pasos_max']))
    partes.append('{} semilla(s)'.format(len(semillas)))
    partes.append('total de {:,.0f} pasos del entorno y {:.1f} h'
                  .format(pasos, horas).replace(',', '.'))
    return ', '.join(partes)


def poner_pie(fig, texto, altura_linea=0.028):
    envuelto = textwrap.fill(texto, width=120)
    lineas = envuelto.count('\n') + 1
    margen = 0.03 + altura_linea * lineas
    fig.text(0.01, 0.01, envuelto, ha='left', va='bottom', fontsize=8.5)
    fig.tight_layout(rect=[0, margen, 1, 1])


def guardar(fig, carpeta, nombre):
    os.makedirs(carpeta, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(carpeta, '{}.{}'.format(nombre, ext)), dpi=200)
    plt.close(fig)


def rango_texto(args):
    if args.q_min is not None and args.q_max is not None:
        return 'Calidad adimensional, rango teorico [{:g}, {:g}].'.format(
            args.q_min, args.q_max)
    return 'Calidad adimensional (rango teorico indicado en el texto).'


# =====================================================================
# Figuras
# =====================================================================

def fig_calidad(semillas, args, ref, num):
    gen = semillas[0]['datos']['generacion'][:apilar(semillas, 'generacion').shape[1]]
    mejor = apilar(semillas, 'cal_mejor')
    media = apilar(semillas, 'cal_media')
    desv = apilar(semillas, 'cal_desv')

    m_mejor, lo_mejor, hi_mejor = agregar(mejor, args.banda)
    m_media = media.mean(axis=0)
    d_pob = desv.mean(axis=0)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    if args.lineas_semilla:
        for fila in mejor:
            ax.plot(gen, fila, color='tab:blue', alpha=0.2, lw=0.8)

    ax.fill_between(gen, lo_mejor, hi_mejor, color='tab:blue', alpha=0.2,
                    label='Banda del mejor ({})'.format(texto_banda(args.banda)))
    ax.plot(gen, m_mejor, color='tab:blue', lw=2, label='Mejor individuo (media de semillas)')

    ax.fill_between(gen, m_media - d_pob, m_media + d_pob, color='tab:orange',
                    alpha=0.2, label='Desviacion tipica de la poblacion')
    ax.plot(gen, m_media, color='tab:orange', lw=2,
            label='Media de la poblacion (media de semillas)')

    if ref is not None:
        ax.axhline(ref, color='k', ls='--', lw=1.3,
                   label='Controlador programado ({:g})'.format(ref))

    ax.set_xlabel('Generacion')
    ax.set_ylabel('Calidad (recompensa acumulada del episodio)')
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right', fontsize=8)

    pie = ('Figura {}. Calidad de la poblacion a lo largo de la evolucion. {} '
           'La banda azul es el {} de la calidad del mejor individuo; la banda naranja es '
           'la desviacion tipica de la poblacion dentro de cada generacion (promediada entre '
           'semillas). {}.'
           .format(num, rango_texto(args), texto_banda(args.banda), datos_pie(semillas)))
    poner_pie(fig, pie)
    guardar(fig, args.salida, 'fig1_calidad')


def matriz_especies(especies, n_gen):
    ids = sorted(especies)
    m = np.zeros((len(ids), n_gen))
    for i, s in enumerate(ids):
        for g, t in especies[s].items():
            if g < n_gen:
                m[i, g] = t
    return ids, m


def fig_especies(semillas, args, num):
    con_datos = [s for s in semillas if s['especies']]
    if not con_datos:
        print('AVISO: no hay datos de especies, se omite la figura {}.'.format(num))
        return
    n = len(con_datos)
    fig, axes = plt.subplots(n, 1, figsize=(9, 2.6 * n + 1.2), sharex=True,
                             squeeze=False)
    cmap = plt.get_cmap('tab20')
    for ax, s in zip(axes[:, 0], con_datos):
        n_gen = len(s['datos']['generacion'])
        ids, m = matriz_especies(s['especies'], n_gen)
        gen = np.arange(n_gen)
        ax.stackplot(gen, m, colors=[cmap(i % 20) for i in range(len(ids))],
                     linewidth=0.3)
        ax.set_ylabel('Individuos\n({})'.format(s['nombre']))
        ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel('Generacion')

    pie = ('Figura {}. Numero y tamano de las especies en cada generacion (areas apiladas; '
           'cada color es una especie y su altura es el numero de individuos que la '
           'forman). Una fila por semilla. {}.'
           .format(num, datos_pie(semillas)))
    poner_pie(fig, pie)
    guardar(fig, args.salida, 'fig2_especies')


def fig_complejidad(semillas, args, num):
    nodos = apilar(semillas, 'nodos_mejor')
    con = apilar(semillas, 'conexiones_mejor')
    gen = np.arange(nodos.shape[1])
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for datos, color, nombre in ((nodos, 'tab:green', 'Nodos'),
                                 (con, 'tab:red', 'Conexiones activas')):
        m, lo, hi = agregar(datos, args.banda)
        ax.fill_between(gen, lo, hi, color=color, alpha=0.2)
        ax.plot(gen, m, color=color, lw=2, label=nombre + ' (media de semillas)')
    ax.set_xlabel('Generacion')
    ax.set_ylabel('Numero de elementos del mejor individuo')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', fontsize=8)

    pie = ('Figura {}. Complejidad del mejor individuo de cada generacion (nodos, incluidos '
           'entradas y salidas, y conexiones activas). La banda es el {}. {}.'
           .format(num, texto_banda(args.banda), datos_pie(semillas)))
    poner_pie(fig, pie)
    guardar(fig, args.salida, 'fig3_complejidad')


# =====================================================================
# Resumen para comentar en el informe
# =====================================================================

def primera_vez(serie, umbral):
    idx = np.where(serie >= umbral)[0]
    return int(idx[0]) if len(idx) else None


def resumen(semillas, ref, args):
    lineas = []
    for s in semillas:
        d = s['datos']
        mejor, media = d['cal_mejor'], d['cal_media']
        hist = np.maximum.accumulate(mejor)
        ultima_mejora = int(np.where(np.diff(hist) > 0)[0][-1] + 1) if np.any(np.diff(hist) > 0) else 0
        n = len(mejor)
        lineas.append('{}: {} generaciones, mejor final {:.2f}, maximo historico {:.2f}, '
                      'media final de la poblacion {:.2f}'
                      .format(s['nombre'], n, mejor[-1], hist[-1], media[-1]))
        lineas.append('    ultima mejora del maximo historico en la generacion {} '
                      '({} generaciones sin mejorar)'.format(ultima_mejora, n - 1 - ultima_mejora))
        if ref is not None:
            gm, gp = primera_vez(mejor, ref), primera_vez(media, ref)
            lineas.append('    el mejor supera la referencia ({:g}) en la generacion {}; '
                          'la media de la poblacion, en la {}'
                          .format(ref, gm if gm is not None else 'nunca',
                                  gp if gp is not None else 'nunca'))
        esp = d['especies']
        lineas.append('    especies: media {:.1f}, maximo {:.0f}, final {:.0f}'
                      .format(esp.mean(), esp.max(), esp[-1]))
        lineas.append('    complejidad final del mejor: {:.0f} nodos, {:.0f} conexiones'
                      .format(d['nodos_mejor'][-1], d['conexiones_mejor'][-1]))
    texto = '\n'.join(lineas)
    print('\n' + texto)
    os.makedirs(args.salida, exist_ok=True)
    with open(os.path.join(args.salida, 'resumen.txt'), 'w') as f:
        f.write(texto + '\n')


# =====================================================================

def main():
    p = argparse.ArgumentParser(description='Curvas de evolucion de NEAT')
    p.add_argument('--carpeta', default='p2_seguir_logs_test')
    p.add_argument('--salida', default='figuras')
    p.add_argument('--referencia', type=float, default=None,
                   help='calidad del controlador a mano (o referencia.json)')
    p.add_argument('--banda', choices=['minmax', 'std'], default='minmax',
                   help='dispersion entre semillas')
    p.add_argument('--q-min', type=float, default=None, help='minimo teorico de la calidad')
    p.add_argument('--q-max', type=float, default=None, help='maximo teorico de la calidad')
    p.add_argument('--lineas-semilla', action='store_true',
                   help='dibuja tambien la curva del mejor de cada semilla')
    p.add_argument('--primera-figura', type=int, default=1,
                   help='numero de la primera figura en los pies')
    args = p.parse_args()

    semillas = cargar_semillas(args.carpeta)
    ref = cargar_referencia(args)
    if len(semillas) < 2:
        print('AVISO: solo hay {} semilla; el enunciado pide al menos dos.'
              .format(len(semillas)))
    if ref is None:
        print('AVISO: no hay calidad de referencia (--referencia o referencia.json).')

    n = args.primera_figura
    fig_calidad(semillas, args, ref, n)
    fig_especies(semillas, args, n + 1)
    fig_complejidad(semillas, args, n + 2)
    resumen(semillas, ref, args)
    print('\nFiguras guardadas en', args.salida)


if __name__ == '__main__':
    main()