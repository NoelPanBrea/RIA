#!/usr/bin/env python3
"""
Simulador en pygame del robot seguidor (pelota verde del guia) + entrenamiento NEAT.

    python3 sim_seguir_neat.py --pista                       # comprobar el camino del guia
    python3 sim_seguir_neat.py --entrenar --generaciones 50  # entrenar sin ventana (rapido)
    python3 sim_seguir_neat.py --entrenar --mostrar-cada 5   # ensenar al mejor cada 5 gen.
    python3 sim_seguir_neat.py --ver                         # ver al mejor genoma guardado

El simulador imita el entorno real (p2_seguir_env.py):
    observacion (7): x blob, tam blob, visible, t sin ver, proximidad IR,
                     v anterior, w anterior
    accion (2, en [-1, 1]): v = (a0+1)/2*V_MAX ; w = a1*W_MAX
"""

import argparse
import math
import os
import pickle
import random
from bisect import bisect_right

import neat
import pygame

AQUI = os.path.dirname(os.path.abspath(__file__))

# =========================
# Parametros del simulador 
# =========================

WIDTH, HEIGHT = 449, 503
PX_POR_M = 167.0         # escala: 20 px (robot) = 0.12 m
RADIO_ROBOT = 10.0       # px
OFFSET_BOLA = 12.0       # px, la bola esta detras del guia, sobre su borde
RADIO_BOLA = 5.0

DT = 0.2                 # segundos simulados por paso (PASO_S + latencia real)
SUBPASOS = 4             # subdivision del paso para detectar choques

# Camara
FOV_MEDIO = math.radians(30)   # semi-angulo de vision horizontal
DIST_MAX_VISION = 1.0          # m, mas lejos no se ve la bola

# IR frontales (FrontLL, FrontL, FrontC, FrontR, FrontRR)
IR_ANGULOS = [math.radians(a) for a in (-60, -30, 0, 30, 60)]
IR_ALCANCE_PX = 25.0

# Ruido de percepcion
RUIDO_X = 0.1
RUIDO_TAM = 0.03

# Guia
V_GUIA_MIN = 0.06   # m/s
V_GUIA_MAX = 0.09   # m/s
ACEL_GUIA = 0.20    # m/s^2
PROB_PARADA = 0

# Camino del guia: rectangulo redondeado por el centro de la carretera
# (izquierda, derecha, arriba, abajo, radio de esquina) en pixeles.
# Se comprueba con:  --pista
LIMITES = (25, 417, 22, 479)
RADIO_ESQUINA = 48

# ---- constantes de p2_seguir_env.py ----
V_MAX = 0.1
W_MAX = 0.3
P_CHOQUE = 0.85
T_PERDIDO_S = 3.0

SIZE_LEJOS = 100
SIZE_CERCA = 250.0
SIZE_MUY_CERCA = 300.0
T_CERCA = 0.12 / 0.20
T_LEJOS = T_CERCA * math.sqrt(SIZE_LEJOS / SIZE_CERCA)
T_MUY_LEJOS = T_LEJOS * 0.45 / 0.90
T_MUY_CERCA = T_CERCA * math.sqrt(SIZE_MUY_CERCA / SIZE_CERCA)

CASTIGO_LEJOS = 0.5
K_SEP = 1.20
K_CENTRO = 0.5
K_NO_VER = 0.5
K_GIRO = 0.1
R_CHOQUE = 10.0
R_PERDIDO = 10.0

RUTA_IMG_PISTA = os.path.join(AQUI, 'AlgoritmoGenéticoroad2.png')
RUTA_IMG_ROBOT = os.path.join(AQUI, 'AlgoritmoGenéticorobobo.png')
RUTA_CONFIG = os.path.join(AQUI, 'config.txt')
RUTA_MODELO = os.path.join(AQUI, 'mejor_genoma.pkl')

CONFIG_NEAT = """[NEAT]
fitness_criterion     = max
fitness_threshold     = 1000000
pop_size              = 100
reset_on_extinction   = True
no_fitness_termination = True

[DefaultGenome]
activation_default      = tanh
activation_mutate_rate  = 0.0
activation_options      = tanh
aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5
conn_add_prob           = 0.5
conn_delete_prob        = 0.5
enabled_default         = True
enabled_mutate_rate     = 0.01
feed_forward            = True
initial_connection      = full_direct
node_add_prob           = 0.2
node_delete_prob        = 0.2
num_hidden              = 0
num_inputs              = 7
num_outputs             = 2
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30
weight_min_value        = -30
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 20
species_elitism      = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2
"""


def angulo_envuelto(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def clip(v, lo, hi):
    return max(lo, min(hi, v))


# =====================================================================
# Pista y camino del guia
# =====================================================================

class Pista:
    def __init__(self, ruta_imagen):
        img = pygame.image.load(ruta_imagen)
        img = pygame.transform.scale(img, (WIDTH, HEIGHT))
        self.imagen = img
        # carretera = todo lo que no es blanco (el verde y la marca roja)
        self.carretera = bytearray(WIDTH * HEIGHT)
        for y in range(HEIGHT):
            for x in range(WIDTH):
                c = img.get_at((x, y))
                blanco = c.r > 200 and c.g > 200 and c.b > 200
                self.carretera[y * WIDTH + x] = 0 if blanco else 1
        self._construir_camino()

    def en_carretera(self, x, y):
        ix, iy = int(x), int(y)
        if ix < 0 or iy < 0 or ix >= WIDTH or iy >= HEIGHT:
            return False
        return self.carretera[iy * WIDTH + ix] == 1

    def _construir_camino(self):
        izq, der, arr, aba = LIMITES
        r = RADIO_ESQUINA
        cx = (izq + der) / 2.0
        pts = []

        def linea(p0, p1):
            n = max(1, int(math.hypot(p1[0] - p0[0], p1[1] - p0[1])))
            for i in range(n):
                t = i / n
                pts.append((p0[0] + (p1[0] - p0[0]) * t,
                            p0[1] + (p1[1] - p0[1]) * t))

        def arco(c, a0, a1):
            n = max(2, int(r * abs(a1 - a0)))
            for i in range(n):
                a = a0 + (a1 - a0) * i / n
                pts.append((c[0] + r * math.cos(a), c[1] + r * math.sin(a)))

        # sentido antihorario en pantalla: abajo hacia la derecha
        linea((cx, aba), (der - r, aba))
        arco((der - r, aba - r), math.pi / 2, 0)
        linea((der, aba - r), (der, arr + r))
        arco((der - r, arr + r), 0, -math.pi / 2)
        linea((der - r, arr), (izq + r, arr))
        arco((izq + r, arr + r), -math.pi / 2, -math.pi)
        linea((izq, arr + r), (izq, aba - r))
        arco((izq + r, aba - r), math.pi, math.pi / 2)
        linea((izq + r, aba), (cx, aba))

        self.puntos = pts
        self.acum = [0.0]
        for i in range(1, len(pts) + 1):
            p0, p1 = pts[i - 1], pts[i % len(pts)]
            self.acum.append(self.acum[-1] + math.hypot(p1[0] - p0[0],
                                                        p1[1] - p0[1]))
        self.longitud = self.acum[-1]

    def punto(self, s):
        s %= self.longitud
        i = bisect_right(self.acum, s) - 1
        i = min(i, len(self.puntos) - 1)
        p0 = self.puntos[i]
        p1 = self.puntos[(i + 1) % len(self.puntos)]
        seg = self.acum[i + 1] - self.acum[i]
        t = (s - self.acum[i]) / seg if seg > 0 else 0.0
        return (p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t)

    def rumbo(self, s):
        a = self.punto(s - 1.0)
        b = self.punto(s + 1.0)
        return math.atan2(b[1] - a[1], b[0] - a[0])

    def fraccion_camino_en_carretera(self):
        ok = sum(1 for p in self.puntos if self.en_carretera(*p))
        return ok / len(self.puntos)


class Guia:
    """Robot guia: recorre el camino con velocidad variable."""

    def __init__(self, pista, rng, s0):
        self.pista = pista
        self.rng = rng
        self.s = s0
        self.v = 0.0
        self.v_obj = 0.0
        self.t_cambio = 0.0

    def avanzar(self, dt):
        self.t_cambio -= dt
        if self.t_cambio <= 0.0:
            if self.rng.random() < PROB_PARADA:
                self.v_obj = 0.0
                self.t_cambio = self.rng.uniform(0.5, 1.5)
            else:
                self.v_obj = self.rng.uniform(V_GUIA_MIN, V_GUIA_MAX)
                self.t_cambio = self.rng.uniform(2.0, 6.0)
        dv = clip(self.v_obj - self.v, -ACEL_GUIA * dt, ACEL_GUIA * dt)
        self.v += dv
        self.s += self.v * PX_POR_M * dt

    @property
    def pos(self):
        return self.pista.punto(self.s)

    @property
    def rumbo(self):
        return self.pista.rumbo(self.s)

    @property
    def bola(self):
        x, y = self.pos
        h = self.rumbo
        return (x - math.cos(h) * OFFSET_BOLA, y - math.sin(h) * OFFSET_BOLA)


# =====================================================================
# Entorno simulado
# =====================================================================

class SimSeguir:
    def __init__(self, pista, pasos_max=400, ruido=True):
        self.pista = pista
        self.pasos_max = pasos_max
        self.ruido = ruido
        self.rng = random.Random()

    # ---------- reset ----------
    def reset(self, seed=None):
        self.rng = random.Random(seed)
        s0 = self.rng.uniform(0, self.pista.longitud)
        d0 = self.rng.uniform(0.22, 0.32) * PX_POR_M + OFFSET_BOLA
        self.guia = Guia(self.pista, self.rng, s0)
        px, py = self.pista.punto(s0 - d0)
        self.x, self.y = px, py
        self.theta = self.pista.rumbo(s0 - d0) + self.rng.gauss(0, 0.1)

        self.pasos = 0
        self.t = 0.0
        self.t_visto = 0.0
        self.ultima_accion = [-1.0, 0.0]
        self._x_ult = 0.0
        self._tam_ult = 0.0
        self.info = {}
        visible, sin_ver, prox = self._percibir()
        return self._observacion(visible, sin_ver, prox)

    # ---------- percepcion ----------
    def _camara(self):
        """Devuelve (visible, x_norm, tam) con la geometria real."""
        bx, by = self.guia.bola
        dx, dy = bx - self.x, by - self.y
        d = max(math.hypot(dx, dy) / PX_POR_M, 0.01)
        ang = angulo_envuelto(math.atan2(dy, dx) - self.theta)
        visible = abs(ang) < FOV_MEDIO and d < DIST_MAX_VISION
        x_norm = ang / FOV_MEDIO              # + a la derecha (y hacia abajo)
        tam = T_CERCA * 0.20 / d              # tam ~ 1/distancia
        if self.ruido:
            x_norm += self.rng.gauss(0, RUIDO_X)
            tam *= 1.0 + self.rng.gauss(0, RUIDO_TAM)
        return visible, clip(x_norm, -1.0, 1.0), min(max(tam, 0.0), 1.0)

    def _proximidad_ir(self):
        gx, gy = self.guia.pos
        oc = (gx - self.x, gy - self.y)
        dist_c = math.hypot(*oc)
        if dist_c < 2 * RADIO_ROBOT:
            return 1.0
        mejor = 0.0
        for da in IR_ANGULOS:
            ux, uy = math.cos(self.theta + da), math.sin(self.theta + da)
            t = oc[0] * ux + oc[1] * uy
            if t < 0:
                continue
            perp2 = dist_c ** 2 - t ** 2
            if perp2 > RADIO_ROBOT ** 2:
                continue
            dist = t - math.sqrt(RADIO_ROBOT ** 2 - perp2)
            hueco = dist - RADIO_ROBOT
            mejor = max(mejor, clip(1.0 - hueco / IR_ALCANCE_PX, 0.0, 1.0))
        return mejor

    def _percibir(self):
        visible, x, tam = self._camara()
        if visible:
            self._x_ult, self._tam_ult = x, tam
            self.t_visto = self.t
        return visible, self.t - self.t_visto, self._proximidad_ir()

    def _observacion(self, visible, sin_ver, prox):
        return [self._x_ult, self._tam_ult, 1.0 if visible else 0.0,
                min(sin_ver / T_PERDIDO_S, 1.0), prox,
                self.ultima_accion[0], self.ultima_accion[1]]

    # ---------- recompensa (identica a la del entorno real) ----------
    def _recompensa(self, visible, x, tam, giro, choque, perdido):
        if visible:
            if tam < T_LEJOS:
                r_sep = max((tam - T_MUY_LEJOS) / (T_LEJOS - T_MUY_LEJOS), 0.0)
                r_sep = r_sep * (1.0 + CASTIGO_LEJOS) - CASTIGO_LEJOS
            elif tam > T_CERCA:
                r_sep = max((T_MUY_CERCA - tam) / (T_MUY_CERCA - T_CERCA), 0.0)
            else:
                r_sep = 1.0
            r = K_SEP * r_sep + K_CENTRO * (1.0 - abs(x))
        else:
            r = -K_NO_VER
        r -= K_GIRO * abs(giro)
        if choque:
            r -= R_CHOQUE
        if perdido:
            r -= R_PERDIDO
        return float(r)

    # ---------- paso ----------
    def step(self, accion):
        a0 = clip(float(accion[0]), -1.0, 1.0)
        a1 = clip(float(accion[1]), -1.0, 1.0)
        v = (a0 + 1.0) * 0.5 * V_MAX
        w = a1 * W_MAX

        h = DT / SUBPASOS
        colision = False
        for _ in range(SUBPASOS):
            self.theta -= w * h   # w>0 = giro a la izquierda (antihorario)
            self.x += math.cos(self.theta) * v * PX_POR_M * h
            self.y += math.sin(self.theta) * v * PX_POR_M * h
            self.guia.avanzar(h)
            gx, gy = self.guia.pos
            if math.hypot(gx - self.x, gy - self.y) < 2 * RADIO_ROBOT:
                colision = True
            if not self.pista.en_carretera(self.x, self.y):
                colision = True
            if colision:
                break
        self.t += DT

        self.pasos += 1
        self.ultima_accion = [a0, a1]
        visible, sin_ver, prox = self._percibir()

        choque = colision or prox >= P_CHOQUE
        perdido = sin_ver > T_PERDIDO_S
        en_banda = visible and T_LEJOS <= self._tam_ult <= T_CERCA

        r = self._recompensa(visible, self._x_ult, self._tam_ult, a1,
                             choque, perdido)
        terminated = bool(choque or perdido)
        truncated = bool(self.pasos >= self.pasos_max)
        self.info = {'visible': visible, 'en_banda': en_banda,
                     'x': self._x_ult, 'tam': self._tam_ult,
                     'proximidad': prox, 'choque': choque,
                     'perdido': perdido, 'v': v, 'w': w}
        return (self._observacion(visible, sin_ver, prox), r,
                terminated, truncated, self.info)


# =====================================================================
# Visualizacion
# =====================================================================

class Visor:
    def __init__(self, pista, ruta_robot, fps=60):
        pygame.init()
        self.pantalla = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('Simulador seguidor - NEAT')
        self.fondo = pista.imagen.convert()
        self.pista = pista
        self.fuente = pygame.font.SysFont('Arial', 14)
        self.reloj = pygame.time.Clock()
        self.fps = fps
        try:
            base = pygame.image.load(ruta_robot).convert_alpha()
            base = pygame.transform.scale(base, (20, 20))
            self.sprite = pygame.transform.rotate(base, 90)
        except Exception:
            self.sprite = None

    def _robot(self, x, y, theta, color):
        if self.sprite is not None:
            rot = pygame.transform.rotate(self.sprite, -math.degrees(theta))
            self.pantalla.blit(rot, rot.get_rect(center=(x, y)))
        else:
            pygame.draw.circle(self.pantalla, (120, 120, 120), (x, y),
                               RADIO_ROBOT)
            pygame.draw.line(self.pantalla, (0, 0, 0), (x, y),
                             (x + RADIO_ROBOT * math.cos(theta),
                              y + RADIO_ROBOT * math.sin(theta)), 2)
        pygame.draw.circle(self.pantalla, color, (int(x), int(y)),
                           int(RADIO_ROBOT) + 2, 2)

    def dibujar(self, sim, texto=''):
        """Devuelve False si se ha cerrado la ventana."""
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False
        self.pantalla.blit(self.fondo, (0, 0))

        # cono de vision del seguidor
        for s in (-1, 1):
            a = sim.theta + s * FOV_MEDIO
            pygame.draw.line(self.pantalla, (100, 50, 255), (sim.x, sim.y),
                             (sim.x + 90 * math.cos(a),
                              sim.y + 90 * math.sin(a)), 1)

        # guia + bola verde
        gx, gy = sim.guia.pos
        self._robot(gx, gy, sim.guia.rumbo, (255, 140, 0))
        bx, by = sim.guia.bola
        pygame.draw.circle(self.pantalla, (0, 160, 0), (int(bx), int(by)),
                           int(RADIO_BOLA))

        # seguidor
        color = (255, 0, 0) if sim.info.get('choque') else (0, 90, 255)
        self._robot(sim.x, sim.y, sim.theta, color)

        i = sim.info
        lineas = [texto,
                  'paso {}  visible {}'.format(sim.pasos,
                                               i.get('visible', '-')),
                  'tam {:.2f}  x {:+.2f}  ir {:.2f}'.format(
                      i.get('tam', 0), i.get('x', 0), i.get('proximidad', 0))]
        for k, l in enumerate(lineas):
            img = self.fuente.render(l, True, (0, 0, 0))
            self.pantalla.blit(img, (WIDTH // 2 - img.get_width() // 2,
                                     180 + 18 * k))
        pygame.display.flip()
        if self.fps > 0:
            self.reloj.tick(self.fps)
        return True


# =====================================================================
# NEAT
# =====================================================================

def jugar_episodio(sim, net, seed, visor=None, texto=''):
    obs = sim.reset(seed)
    total = 0.0
    while True:
        accion = net.activate(obs)          # tanh -> [-1, 1]
        obs, r, term, trunc, info = sim.step(accion)
        total += r
        if visor is not None and not visor.dibujar(sim, texto):
            return None, info
        if term or trunc:
            return total, info


def cargar_config(ruta):
    if not os.path.exists(ruta):
        with open(ruta, 'w') as f:
            f.write(CONFIG_NEAT)
        print('Creado {} con la configuracion por defecto.'.format(ruta))
    return neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                              neat.DefaultSpeciesSet, neat.DefaultStagnation,
                              ruta)


def modo_pista(args, pista):
    print('Longitud del camino: {:.0f} px'.format(pista.longitud))
    print('Camino sobre la carretera: {:.1f} %'.format(
        100 * pista.fraccion_camino_en_carretera()))
    visor = Visor(pista, args.imagen_robot, fps=30)
    visor.pantalla.blit(visor.fondo, (0, 0))
    for p in pista.puntos:
        visor.pantalla.set_at((int(p[0]), int(p[1])), (255, 0, 255))
    pygame.display.flip()
    print('Cierra la ventana para salir.')
    while not any(e.type == pygame.QUIT for e in pygame.event.get()):
        pygame.time.wait(50)


def modo_entrenar(args, pista):
    config = cargar_config(args.config)
    sim = SimSeguir(pista, pasos_max=args.pasos_max)
    visor = Visor(pista, args.imagen_robot, args.fps) \
        if args.mostrar_cada > 0 else None
    estado = {'gen': 0, 'mejor': None, 'mejor_fit': -1e18}

    def evaluar(genomes, cfg):
        estado['gen'] += 1
        seeds = [estado['gen'] * 1000 + k for k in range(args.episodios)]
        mejor_gen = None
        for _, g in genomes:
            net = neat.nn.FeedForwardNetwork.create(g, cfg)
            tot = [jugar_episodio(sim, net, s)[0] for s in seeds]
            g.fitness = sum(tot) / len(tot)
            if mejor_gen is None or g.fitness > mejor_gen.fitness:
                mejor_gen = g
        if mejor_gen.fitness > estado['mejor_fit']:
            estado['mejor_fit'] = mejor_gen.fitness
            estado['mejor'] = mejor_gen
            with open(args.modelo, 'wb') as f:
                pickle.dump(mejor_gen, f)
        print('  mejor de la generacion {}: {:.2f}   (record {:.2f})'.format(
            estado['gen'], mejor_gen.fitness, estado['mejor_fit']))

        if visor and estado['gen'] % args.mostrar_cada == 0:
            net = neat.nn.FeedForwardNetwork.create(mejor_gen, cfg)
            jugar_episodio(sim, net, seeds[0], visor,
                           'Gen {}  fit {:.1f}'.format(estado['gen'],
                                                       mejor_gen.fitness))

    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.StatisticsReporter())
    winner = pop.run(evaluar, args.generaciones)
    with open(args.modelo, 'wb') as f:
        pickle.dump(winner, f)
    print('\nGenoma guardado en', args.modelo)


def modo_ver(args, pista):
    config = cargar_config(args.config)
    with open(args.modelo, 'rb') as f:
        genoma = pickle.load(f)
    net = neat.nn.FeedForwardNetwork.create(genoma, config)
    sim = SimSeguir(pista, pasos_max=args.pasos_max)
    visor = Visor(pista, args.imagen_robot, args.fps)
    seed = 0
    while True:
        seed += 1
        total, info = jugar_episodio(sim, net, seed, visor,
                                     'Episodio {}'.format(seed))
        if total is None:
            break
        fin = 'choque' if info['choque'] else \
            ('guia perdido' if info['perdido'] else 'tiempo agotado')
        print('episodio {:3d}  recompensa {:8.2f}  {}'.format(seed, total, fin))


def main():
    p = argparse.ArgumentParser(description='Simulador NEAT del seguidor')
    m = p.add_mutually_exclusive_group(required=True)
    m.add_argument('--pista', action='store_true')
    m.add_argument('--entrenar', action='store_true')
    m.add_argument('--ver', action='store_true')
    p.add_argument('--generaciones', type=int, default=50)
    p.add_argument('--episodios', type=int, default=3,
                   help='episodios por genoma (se promedia)')
    p.add_argument('--pasos-max', type=int, default=400)
    p.add_argument('--mostrar-cada', type=int, default=0,
                   help='ensenar al mejor cada N generaciones (0 = nunca)')
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--config', default=RUTA_CONFIG)
    p.add_argument('--modelo', default=RUTA_MODELO)
    p.add_argument('--imagen-pista', default=RUTA_IMG_PISTA)
    p.add_argument('--imagen-robot', default=RUTA_IMG_ROBOT)
    args = p.parse_args()

    pista = Pista(args.imagen_pista)
    if args.pista:
        modo_pista(args, pista)
    elif args.entrenar:
        modo_entrenar(args, pista)
    else:
        modo_ver(args, pista)


if __name__ == '__main__':
    main()