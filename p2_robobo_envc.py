#!/usr/bin/env python3
"""
Práctica 1 — Entorno Gymnasium: el Robobo SEGUIDOR persigue al VEHÍCULO GUÍA
por el anillo exterior de la ciudad de RoboboSim.

Este archivo contiene únicamente el entorno y las utilidades de medida
(calibración del sensor y grabación de la línea central del carril).
El entrenamiento y el evaluador del anexo B van en otros archivos.

--------------------------------------------------------------------
ESTRUCTURA
--------------------------------------------------------------------
  base/, smartphone/   percepción y actuación del seguidor. Es lo único
                       que existe también en el robot real.
  sim/                 reinicio de la escena, colocación de robots y
                       posiciones reales. Información PRIVILEGIADA: sirve
                       para reiniciar, calcular la recompensa, decidir
                       cuándo termina el episodio y registrar métricas,
                       pero NUNCA entra en la observación (apartado 4).

--------------------------------------------------------------------
REQUISITOS
--------------------------------------------------------------------
1. RoboboSim en ejecución, con el escenario de la ciudad y el guía
   circulando (modo y velocidades configurados en el simulador).

2. Contenedor del SEGUIDOR (módulos blob y sim):

     ros2 run robobo_ros2 robobo_container --ros-args \
          -p ip:=host.docker.internal -p modules:="['blob','sim']"

3. Contenedor del GUÍA, solo con el módulo sim (anexo A), con el índice
   del otro robot, para conocer su posición real. Ajustar 'ns_guia' en
   el archivo de configuración si su espacio de nombres no es
   /robobo/robot_1.

4. Solo si modo_recompensa='verdad' (y para el evaluador del anexo B):
   línea central del carril, generada una vez con grabar_linea_central().
   En el modo por defecto, 'sensores', NO hace falta.

--------------------------------------------------------------------
CALIBRACIÓN
--------------------------------------------------------------------
Todo lo que depende del simulador o del robot real (normalización de la
imagen, filtro de blobs, límites de velocidad, geometría, coeficientes de
la recompensa...) está en un archivo JSON de configuración y NO en el
código (apartado 5.5). Si el archivo no existe se usan los valores de
CONFIG_DEFECTO. Procedimiento: ejecutar calibrar() en simulación y en el
robot real, y ajustar el archivo con las lecturas.

--------------------------------------------------------------------
SUPUESTOS SOBRE LAS INTERFACES (comprobar con `ros2 interface show`)
--------------------------------------------------------------------
  * robobo_ros2_interfaces/msg/BlobArray tiene un campo `blobs` y cada
    blob tiene `color`, `x`, `y`, `size`.
  * robobo_ros2_interfaces/srv/SetRobotLocation tiene `position` y
    `rotation` (Vector3) en la petición.
  * Yaw del simulador: grados alrededor de y, con 0° mirando a +z
    (dirección = (sin yaw, cos yaw) en el plano (x, z)).
"""

import copy
import csv
import json
import math
import os
import time

import numpy as np

import gymnasium as gym
from gymnasium import spaces

import rclpy
from rclpy.node import Node

from std_msgs.msg import Int32MultiArray
from geometry_msgs.msg import Twist

# Interfaces propias del puente. Las de sim no existen en el robot real.
from robobo_ros2_interfaces.msg import BlobArray, RobotLocation
from robobo_ros2_interfaces.srv import ResetSimulation, SetRobotLocation


# =====================================================================
# Configuración (archivo JSON; estos son los valores por defecto)
# =====================================================================

CONFIG_DEFECTO = {
    # ---- espacios de nombres (valores por defecto del nodo puente)
    'ns_robot': '/robobo/robot_0',        # seguidor
    'ns_guia': '/robobo/robot_1',         # contenedor sim del guía

    # ---- percepción: blob de la pelota verde
    'color_blob': 'green',
    'blob_x_max': 100.0,       # rango de x en la imagen (CALIBRAR)
    'blob_y_max': 100.0,       # rango de y en la imagen (CALIBRAR)
    'blob_size_max': 3000.0,   # tamaño con el guía pegado (CALIBRAR)
    'blob_size_min': 20.0,     # por debajo: ruido u otro elemento verde
    'blob_y_validas': [0.0, 100.0],   # franja vertical donde puede estar
                                      # la pelota; fuera: otro verde
    'ir_saturacion': 1000.0,   # valor bruto IR al tocar (CALIBRAR)
    'usar_ir': True,           # variante de observación: añade IR frontal
    'usar_delta': True,        # variante: añade variación del tamaño
    'escala_delta': 5.0,
    'sin_ver_max': 25,         # pasos sin ver la pelota que satura el contador

    # ---- actuación (cmd_vel). Límites físicos (CALIBRAR)
    'v_max': 0.25,             # m/s
    'v_min': -0.08,            # m/s
    'w_max': 1.0,              # rad/s

    # ---- temporización
    # Los blobs se publican a 5 Hz (tiempo real). La acción se aplica
    # 'espera_accion_s' y después se espera la siguiente lectura, por lo
    # que el periodo de control efectivo es ~0.2 s (se registra en
    # info['dt']). Debe ser < cmd_vel_timeout del puente (0.5 s).
    'espera_accion_s': 0.1,
    'espera_max_s': 3.0,
    'pasos_max': 300,          # 60 s / 0.2 s

    # ---- geometría del anexo B (metros)
    'banda_sep': [0.25, 0.50],
    'sep_perdido': 1.0,
    'd_colision': 0.20,
    'ancho_carril': 0.22,
    'ancho_robot': 0.16,
    # Límites del error lateral con signo (+ = hacia el lado definido por
    # el producto vectorial tangente x desplazamiento en el plano x-z)
    # que delimitan la calzada: carril propio (±0.11) + carril contrario
    # (0.22 más hacia un lado). CALIBRAR el signo del lado contrario.
    'calzada_lat': [-0.11, 0.33],
    'linea_central': 'linea_central.json',

    # Origen de la recompensa y de las terminaciones:
    #   'sensores': SOLO lo que mide el robot (blob + IR). No usa la línea
    #               central ni la pose del guía. Funciona igual en el robot
    #               real. El simulador solo se usa para reiniciar.
    #   'verdad'  : usa la verdad del terreno del simulador (anexo B);
    #               requiere linea_central.json.
    'modo_recompensa': 'sensores',
    'tam_banda': [0.35, 0.60],   # banda objetivo en 'tam' (CALIBRAR: valor
                                 # de tam a 0.50 m y a 0.25 m de separación)
    'ir_choque': 0.85,           # proximidad IR frontal que se da por choque
    'sin_ver_fin': 15,           # pasos sin ver la pelota = guía perdido
    # El guía solo arranca cuando el seguidor se acerca: hasta que se vea
    # la pelota por primera vez no se da por perdido, con un máximo de
    # 'pasos_gracia' pasos (50 pasos ~ 10 s) para localizarlo.
    'pasos_gracia': 50,
    # False si no se puede leer la pose del guía (contenedor sim del guía
    # no lanzado). Entonces el reinicio solo perturba la pose del propio
    # seguidor.
    'guia_pose_disponible': False,

    # ---- coeficientes de la recompensa (valor numérico de cada término)
    'recompensa': {
        'sep_banda': 1.0,     # + por paso dentro de la banda objetivo
        'sep_dist': 2.0,      # - por metro de distancia a la banda (<=1 m)
        'carril': 1.0,        # - por |lat|/ancho_carril (saturado a 1)
        'sin_vista': 0.2,     # - por paso sin ver la pelota
        'giro': 0.05,         # - por |w|/w_max
        'cambio_accion': 0.05,# - por ||a_t - a_{t-1}|| / 2
        'terminal': 50.0,     # - por colisión, salida de calzada o guía perdido
        # solo modo 'sensores':
        'tam_dist': 4.0,      # - por unidad de 'tam' fuera de la banda
        'centrado': 0.5,      # - por |x| del blob (pelota centrada)
    },

    # ---- reinicio de episodios (solo entrenamiento)
    'aleatorizar_inicio': True,     # colocar al seguidor detrás del guía
    'inicio_sep': [0.30, 0.70],     # separación inicial (m)
    'inicio_lat_sigma': 0.02,       # ruido lateral (m)
    'inicio_yaw_sigma_deg': 10.0,   # ruido de orientación (grados)
    'aleatorizar_guia': False,      # además, mover al guía a otro punto
                                    # del anillo (puede interferir con su
                                    # controlador: probar antes)

    # ---- sin simulador (robot real): el guía no se puede medir
    'max_pasos_sin_blob_real': 25,
}


def cargar_config(config=None):
    """Devuelve la configuración completa.

    `config` puede ser None (valores por defecto), una ruta a un JSON o
    un dict. Las claves ausentes se rellenan con CONFIG_DEFECTO.
    """
    cfg = copy.deepcopy(CONFIG_DEFECTO)
    if isinstance(config, str):
        if os.path.exists(config):
            with open(config) as f:
                config = json.load(f)
        else:
            print('[config] {} no existe; se usan los valores por defecto'
                  .format(config))
            config = None
    if isinstance(config, dict):
        for k, v in config.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg


def guardar_config(ruta, config=None):
    """Escribe la configuración (por defecto o la dada) en un JSON."""
    with open(ruta, 'w') as f:
        json.dump(cargar_config(config), f, indent=2)


# =====================================================================
# Geometría: línea central del carril
# =====================================================================

class LineaCentral:
    """Polilínea cerrada (metros, plano x-z) que describe el guía.

    Permite proyectar un punto y obtener (s, lat, dist): posición a lo
    largo de la línea, error lateral con signo y distancia mínima.
    El sentido de s es el de marcha del guía.
    """

    def __init__(self, puntos):
        P = np.asarray(puntos, dtype=float)
        self.P = P
        self.Q = np.roll(P, -1, axis=0)             # fin de cada segmento
        self.d = self.Q - self.P
        self.largo = np.linalg.norm(self.d, axis=1)
        self.largo[self.largo < 1e-9] = 1e-9
        self.tang = self.d / self.largo[:, None]
        self.s0 = np.concatenate([[0.0], np.cumsum(self.largo)[:-1]])
        self.L = float(self.largo.sum())

    @classmethod
    def desde_archivo(cls, ruta):
        with open(ruta) as f:
            return cls(json.load(f)['puntos_m'])

    def proyectar(self, x, z):
        p = np.array([x, z], dtype=float)
        t = np.clip(np.einsum('ij,ij->i', p - self.P, self.d)
                    / self.largo ** 2, 0.0, 1.0)
        c = self.P + self.d * t[:, None]
        dist = np.linalg.norm(p - c, axis=1)
        i = int(np.argmin(dist))
        tx, tz = self.tang[i]
        dx, dz = p - c[i]
        lat = tx * dz - tz * dx                      # con signo
        return float(self.s0[i] + t[i] * self.largo[i]), float(lat), float(dist[i])

    def punto(self, s):
        """(x, z, tx, tz) de la línea en la posición s."""
        s = s % self.L
        i = int(np.searchsorted(self.s0, s, side='right') - 1)
        t = (s - self.s0[i]) / self.largo[i]
        x, z = self.P[i] + self.d[i] * t
        return float(x), float(z), float(self.tang[i][0]), float(self.tang[i][1])

    def separacion(self, s_guia, s_seguidor):
        """Diferencia de arco guía - seguidor, en (-L/2, L/2]. Negativa si
        el seguidor va delante."""
        d = (s_guia - s_seguidor) % self.L
        return d - self.L if d > self.L / 2 else d


# =====================================================================
# Utilidades de mensajes
# =====================================================================

def extraer_blob(msg, color):
    """Devuelve (x, y, size) del blob de ese color, o None.

    Si hubiese varios del mismo color se toma el mayor.
    """
    mejor = None
    for b in getattr(msg, 'blobs', []):
        if str(b.color).lower() != color:
            continue
        if mejor is None or b.size > mejor[2]:
            mejor = (float(b.x), float(b.y), float(b.size))
    return mejor


def _pose_de_msg(msg):
    """RobotLocation (mm, grados) -> dict en metros."""
    return {'x': msg.position.x / 1000.0, 'y': msg.position.y,
            'z': msg.position.z / 1000.0, 'yaw': msg.rotation.y,
            't': time.time()}


# =====================================================================
# El entorno
# =====================================================================

class RoboboSeguidorEnv(gym.Env):
    """Seguimiento de un vehículo guía con el Robobo en RoboboSim.

    Observación (Box, float32). Solo sensores del seguidor:
        visible      1 si se ve la pelota verde, 0 si no
        x            posición horizontal del blob, en [-1, 1] (0 si no se ve)
        y            posición vertical del blob, en [-1, 1] (0 si no se ve)
        tam          sqrt(size / size_max), en [0, 1] (0 si no se ve).
                     El diámetro aparente es ~ 1/distancia, de ahí la raíz.
        d_tam        variación de tam respecto al paso anterior (solo si
                     usar_delta): informa de si el guía se aleja o se acerca
        x_ult        última x vista (memoria: hacia qué lado se perdió)
        sin_ver      pasos sin ver la pelota / sin_ver_max, en [0, 1]
        ir_front     proximidad IR frontal máxima, en [0, 1] (solo si
                     usar_ir): cubre la zona ciega de la cámara
        v_prev, w_prev  acción anterior en [-1, 1]

    Acción (Box, float32, dimensión 2), continua
        [0] velocidad lineal en [-1, 1] -> [v_min, v_max] (m/s)
        [1] velocidad angular en [-1, 1] -> [-w_max, w_max] (rad/s)
        Se envía por cmd_vel (unidades del SI, cinemática resuelta por el
        puente y parada de seguridad si no llegan órdenes).

    Terminación (con verdad del terreno, criterios del anexo B)
        colisión, fuera de la calzada, guía perdido.
    Truncamiento
        pasos_max pasos (60 s).
    """

    metadata = {'render_modes': []}

    def __init__(self, config=None, usar_simulador=True,
                 modo_evaluacion=False, verbose=False):
        super().__init__()

        self.cfg = cargar_config(config)
        self.usar_sim = usar_simulador
        # Sin simulador no hay verdad del terreno: siempre 'sensores'.
        self.modo = self.cfg['modo_recompensa'] if usar_simulador else 'sensores'
        self.cfg['modo_recompensa'] = self.modo
        self.modo_eval = modo_evaluacion     # sin aleatorización de inicio
        self.verbose = verbose
        c = self.cfg

        self.ns_base = c['ns_robot'] + '/base'
        self.ns_phone = c['ns_robot'] + '/smartphone'
        self.ns_sim = c['ns_robot'] + '/sim'
        self.ns_sim_guia = c['ns_guia'] + '/sim'

        # -------------------------------------------------- ROS 2
        if not rclpy.ok():
            rclpy.init()
        self.nodo = Node('p1_entorno_rl')

        # Publicador y suscripciones se crean UNA vez, aquí.
        self.pub_vel = self.nodo.create_publisher(
            Twist, self.ns_base + '/cmd_vel', 10)

        self._blobs = None
        self._n_blob = 0
        self.nodo.create_subscription(
            BlobArray, self.ns_phone + '/color_blobs', self._cb_blobs, 1)

        self._ir = None
        if c['usar_ir']:
            self.nodo.create_subscription(
                Int32MultiArray, self.ns_base + '/ir', self._cb_ir, 1)

        # -------------------------------------------------- simulador
        self.cli_reset = None
        self.cli_set = None
        self.cli_set_guia = None
        self._pose = None        # seguidor (verdad del terreno)
        self._pose_guia = None   # guía (verdad del terreno)
        self.linea = None
        if usar_simulador:
            self.cli_reset = self.nodo.create_client(
                ResetSimulation, self.ns_sim + '/reset_simulation')
            self.cli_set = self.nodo.create_client(
                SetRobotLocation, self.ns_sim + '/set_robot_location')
            if not self.cli_reset.wait_for_service(timeout_sec=5.0):
                raise RuntimeError(
                    'No responde {}/reset_simulation.\nLanzar robobo_container '
                    'con el módulo sim, o usar usar_simulador=False.'
                    .format(self.ns_sim))
            self.nodo.create_subscription(
                RobotLocation, self.ns_sim + '/robot_location',
                self._cb_pose, 1)
            if c['guia_pose_disponible'] or c['modo_recompensa'] == 'verdad':
                self.nodo.create_subscription(
                    RobotLocation, self.ns_sim_guia + '/robot_location',
                    self._cb_pose_guia, 1)
            if c['aleatorizar_guia']:
                self.cli_set_guia = self.nodo.create_client(
                    SetRobotLocation, self.ns_sim_guia + '/set_robot_location')
            if c['modo_recompensa'] == 'verdad':
                if not os.path.exists(c['linea_central']):
                    raise RuntimeError(
                        'Falta la línea central ({}). Generarla con '
                        'grabar_linea_central() o usar modo_recompensa='
                        '"sensores".'.format(c['linea_central']))
                self.linea = LineaCentral.desde_archivo(c['linea_central'])

        # -------------------------------------------------- espacios
        n_obs = 7 + int(c['usar_delta']) + int(c['usar_ir'])
        # visible, x, y, tam, [d_tam], x_ult, sin_ver, [ir], v_prev, w_prev
        low = [0.0, -1.0, -1.0, 0.0]
        high = [1.0, 1.0, 1.0, 1.0]
        if c['usar_delta']:
            low.append(-1.0); high.append(1.0)
        low += [-1.0, 0.0]; high += [1.0, 1.0]
        if c['usar_ir']:
            low.append(0.0); high.append(1.0)
        low += [-1.0, -1.0]; high += [1.0, 1.0]
        assert len(low) == n_obs
        self.observation_space = spaces.Box(
            low=np.array(low, dtype=np.float32),
            high=np.array(high, dtype=np.float32), dtype=np.float32)

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # -------------------------------------------------- estado
        self.pasos = 0
        self.ultima_accion = np.zeros(2, dtype=np.float32)
        self._cmd = (0.0, 0.0)
        self._t_pub = 0.0
        self._t_ultimo_paso = time.time()
        self._reset_percepcion()

        # Confirma que el puente publica antes de empezar a entrenar.
        self._esperar_lectura_nueva(inicial=True)

    # -----------------------------------------------------------------
    # Comunicación con ROS 2
    # -----------------------------------------------------------------

    def _cb_blobs(self, msg):
        """Se ejecuta dentro de spin_once(); nunca por su cuenta."""
        self._blobs = msg
        self._n_blob += 1

    def _cb_ir(self, msg):
        self._ir = list(msg.data)

    def _cb_pose(self, msg):
        self._pose = _pose_de_msg(msg)

    def _cb_pose_guia(self, msg):
        self._pose_guia = _pose_de_msg(msg)

    def _vaciar_cola(self):
        """Descarta mensajes pendientes: son anteriores o simultáneos a la
        acción y no reflejan su efecto completo."""
        for _ in range(100):
            antes = self._n_blob
            rclpy.spin_once(self.nodo, timeout_sec=0.0)
            if self._n_blob == antes:
                return

    def _esperar_lectura_nueva(self, inicial=False):
        """Bloquea hasta recibir un mensaje de blobs POSTERIOR a este
        instante.

        Garantía del requisito 5.2: step() siempre hace
          1) publicar la acción;  2) dormir espera_accion_s sin hacer spin;
          3) _vaciar_cola() (se descarta todo lo publicado antes de
             ahora);  4) esperar un mensaje NUEVO.
        El mensaje que se consume en 4) se publicó después de 3), es
        decir, al menos espera_accion_s después de aplicar la acción. Los
        blobs se publican siempre (también vacíos) a 5 Hz, así que la
        espera acaba en <= 0.2 s. Mientras se espera se reenvía la orden
        para que no salte el temporizador de seguridad de cmd_vel.
        """
        objetivo = self._n_blob + 1
        t0 = time.time()
        while self._n_blob < objetivo:
            rclpy.spin_once(self.nodo, timeout_sec=0.02)
            if time.time() - self._t_pub > 0.25 and self._cmd != (0.0, 0.0):
                self._publicar_velocidad(*self._cmd)
            if time.time() - t0 > self.cfg['espera_max_s']:
                raise RuntimeError(
                    'No llegan lecturas de {}/color_blobs.\nComprobar el '
                    'nodo puente y el módulo blob.'.format(self.ns_phone)
                    if inicial else
                    'Se ha interrumpido el flujo de blobs durante el episodio.')

    def _publicar_velocidad(self, v, w):
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.pub_vel.publish(msg)
        self._cmd = (float(v), float(w))
        self._t_pub = time.time()

    def _detener(self):
        for _ in range(3):
            self._publicar_velocidad(0.0, 0.0)
            time.sleep(0.05)

    # -----------------------------------------------------------------
    # Simulador: reinicio y colocación
    # -----------------------------------------------------------------

    def _reiniciar_escena(self):
        futuro = self.cli_reset.call_async(ResetSimulation.Request())
        rclpy.spin_until_future_complete(self.nodo, futuro, timeout_sec=5.0)
        if futuro.result() is None or not futuro.result().success:
            raise RuntimeError('El reinicio de la escena ha fallado.')

    def _colocar(self, cliente, x_m, y, z_m, yaw_deg):
        req = SetRobotLocation.Request()
        req.position.x = float(x_m * 1000.0)
        req.position.y = float(y)
        req.position.z = float(z_m * 1000.0)
        req.rotation.x = 0.0
        req.rotation.y = float(yaw_deg)
        req.rotation.z = 0.0
        futuro = cliente.call_async(req)
        rclpy.spin_until_future_complete(self.nodo, futuro, timeout_sec=5.0)
        if futuro.result() is None:
            raise RuntimeError('No se ha podido colocar el robot.')

    def _esperar_poses(self, requiere_guia=True):
        """Espera a tener pose reciente del seguidor (y del guía)."""
        t0 = time.time()
        self._pose = None
        self._pose_guia = None
        while self._pose is None or (requiere_guia and self._pose_guia is None):
            rclpy.spin_once(self.nodo, timeout_sec=0.05)
            if time.time() - t0 > self.cfg['espera_max_s']:
                raise RuntimeError(
                    'No llegan las posiciones del simulador. ¿Está lanzado '
                    'el contenedor sim del guía ({})?'.format(self.ns_sim_guia))

    def _colocar_inicio(self):
        """Coloca al seguidor sobre la línea central, detrás del guía.

        Los episodios de entrenamiento empiezan con separación, desviación
        lateral y orientación aleatorias: así la política no puede
        memorizar una secuencia fija de acciones. (Con aleatorizar_guia,
        el guía también se coloca en un punto aleatorio del anillo, para
        ver rectas y curvas desde el inicio.)
        """
        c = self.cfg
        rng = self.np_random
        if not c['guia_pose_disponible'] and self.modo != 'verdad':
            # Sin pose del guía: se perturba la pose inicial del propio
            # seguidor (posición y orientación) para que no todos los
            # episodios empiecen idénticos.
            self._esperar_poses(requiere_guia=False)
            p = self._pose
            self._colocar(
                self.cli_set,
                p['x'] + float(rng.normal(0.0, c['inicio_lat_sigma'])),
                p['y'],
                p['z'] + float(rng.normal(0.0, c['inicio_lat_sigma'])),
                p['yaw'] + float(rng.normal(0.0, c['inicio_yaw_sigma_deg'])))
            return
        self._esperar_poses()
        g = self._pose_guia
        if self.linea is None:
            # Sin línea central: detrás del guía, en la dirección de su
            # orientación (exacto en rectas; aproximado en curvas).
            sep0 = float(rng.uniform(*c['inicio_sep']))
            a = math.radians(g['yaw'])
            lat = float(rng.normal(0.0, c['inicio_lat_sigma']))
            x = g['x'] - sep0 * math.sin(a) - lat * math.cos(a)
            z = g['z'] - sep0 * math.cos(a) + lat * math.sin(a)
            yaw = g['yaw'] + float(rng.normal(0.0, c['inicio_yaw_sigma_deg']))
            self._colocar(self.cli_set, x, self._pose['y'], z, yaw)
            return
        s_g, _, _ = self.linea.proyectar(g['x'], g['z'])

        if self.cli_set_guia is not None:
            s_g = float(rng.uniform(0, self.linea.L))
            x, z, tx, tz = self.linea.punto(s_g)
            self._colocar(self.cli_set_guia, x, g['y'], z,
                          math.degrees(math.atan2(tx, tz)))

        sep0 = float(rng.uniform(*c['inicio_sep']))
        x, z, tx, tz = self.linea.punto(s_g - sep0)
        lat = float(rng.normal(0.0, c['inicio_lat_sigma']))
        x += -tz * lat           # normal coherente con la definición de lat
        z += tx * lat
        yaw = (math.degrees(math.atan2(tx, tz))
               + float(rng.normal(0.0, c['inicio_yaw_sigma_deg'])))
        self._colocar(self.cli_set, x, self._pose['y'], z, yaw)

    # -----------------------------------------------------------------
    # Percepción
    # -----------------------------------------------------------------

    def _reset_percepcion(self):
        self._visto = False      # ¿se ha visto la pelota en este episodio?
        self._visible = False
        self._x = 0.0
        self._y = 0.0
        self._tam = 0.0
        self._d_tam = 0.0
        self._x_ult = 0.0
        self._sin_ver = 0
        self._ir_front = 0.0

    def _actualizar_percepcion(self):
        """Procesa el último mensaje de blobs (y los IR).

        Filtro de "un blob por color": el detector entrega un único blob
        verde por imagen y en la ciudad hay otros elementos verdes (el
        césped). Se descarta el blob si es demasiado pequeño o está fuera
        de la franja vertical donde puede estar la pelota.
        """
        c = self.cfg
        b = extraer_blob(self._blobs, c['color_blob']) if self._blobs else None
        if b is not None:
            x, y, size = b
            ymin, ymax = c['blob_y_validas']
            if size < c['blob_size_min'] or not (ymin <= y <= ymax):
                b = None

        tam_ant = self._tam if self._visible else None
        if b is None:
            self._visible = False
            self._x = self._y = self._tam = 0.0
            self._d_tam = 0.0
            self._sin_ver += 1
        else:
            x, y, size = b
            self._visible = True
            self._visto = True
            self._x = float(np.clip(2.0 * x / c['blob_x_max'] - 1.0, -1, 1))
            self._y = float(np.clip(2.0 * y / c['blob_y_max'] - 1.0, -1, 1))
            self._tam = float(np.clip(
                math.sqrt(max(size, 0.0) / c['blob_size_max']), 0.0, 1.0))
            self._d_tam = 0.0 if tam_ant is None else float(np.clip(
                (self._tam - tam_ant) * c['escala_delta'], -1.0, 1.0))
            self._x_ult = self._x
            self._sin_ver = 0

        if c['usar_ir'] and self._ir is not None:
            frontales = [self._ir[i] for i in range(5)]   # FrontLL..FrontRR
            self._ir_front = float(max(min(max(v, 0) / c['ir_saturacion'], 1.0)
                                       for v in frontales))

    def _observacion(self):
        c = self.cfg
        o = [1.0 if self._visible else 0.0, self._x, self._y, self._tam]
        if c['usar_delta']:
            o.append(self._d_tam)
        o += [self._x_ult, min(self._sin_ver / c['sin_ver_max'], 1.0)]
        if c['usar_ir']:
            o.append(self._ir_front)
        o += list(self.ultima_accion)
        return np.array(o, dtype=np.float32)

    # -----------------------------------------------------------------
    # Verdad del terreno (anexo B). NO entra en la observación.
    # -----------------------------------------------------------------

    def _verdad(self):
        """(sep, lat, d_centros, s_seguidor) o None si no hay simulador."""
        if (not self.usar_sim or self.linea is None or self._pose is None
                or self._pose_guia is None):
            return None
        f, g = self._pose, self._pose_guia
        s_f, lat, _ = self.linea.proyectar(f['x'], f['z'])
        s_g, _, _ = self.linea.proyectar(g['x'], g['z'])
        sep = self.linea.separacion(s_g, s_f)
        d = math.hypot(f['x'] - g['x'], f['z'] - g['z'])
        return sep, lat, d

    def _causa_fin(self, sep, lat, d):
        c = self.cfg
        if d < c['d_colision']:
            return 'colision'
        if not (c['calzada_lat'][0] <= lat <= c['calzada_lat'][1]):
            return 'fuera_calzada'
        if sep > c['sep_perdido'] or sep < 0.0:
            return 'guia_perdido'
        return None

    # -----------------------------------------------------------------
    # Recompensa
    # -----------------------------------------------------------------

    def _recompensa(self, sep, lat, w, accion, causa):
        """Recompensa del paso, con cada término por separado.

        Recoge los cuatro objetivos del apartado 1:
          - separación en la banda objetivo : + sep_banda dentro de la
            banda; fuera, - sep_dist por metro de distancia a la banda.
          - sin chocar / sin perder al guía / sin salirse de la calzada:
            penalización terminal (- terminal) al terminar el episodio por
            cualquiera de esas causas; como cada paso dentro de la banda
            suma ~+1 y el episodio dura 300 pasos, perder el episodio
            cuesta más que cualquier ganancia por acortarlo.
          - mantenerse en su carril: - carril * min(|lat|/ancho_carril, 1)
            en cada paso (señal densa de error lateral, también en curvas).
          - Regularización: - sin_vista si no se ve la pelota, - giro por
            giro excesivo, - cambio_accion por cambios bruscos de acción.
        Devuelve (total, dict de términos).
        """
        k = self.cfg['recompensa']
        lo, hi = self.cfg['banda_sep']
        dist_banda = max(0.0, lo - sep, sep - hi)

        t = {}
        t['sep'] = (k['sep_banda'] if dist_banda == 0.0
                    else -k['sep_dist'] * min(dist_banda, 1.0))
        t['carril'] = -k['carril'] * min(abs(lat) / self.cfg['ancho_carril'], 1.0)
        t['sin_vista'] = 0.0 if self._visible else -k['sin_vista']
        t['giro'] = -k['giro'] * abs(w) / self.cfg['w_max']
        t['cambio'] = -k['cambio_accion'] * float(
            np.linalg.norm(accion - self._accion_prev)) / 2.0
        t['terminal'] = -k['terminal'] if causa in (
            'colision', 'fuera_calzada', 'guia_perdido') else 0.0
        return float(sum(t.values())), t

    # -----------------------------------------------------------------
    # Interfaz de Gymnasium
    # -----------------------------------------------------------------

    def _recompensa_sensores(self, w, accion, causa):
        """Recompensa calculada SOLO con lo que mide el robot.

          - separación: 'tam' (tamaño aparente de la pelota, ~1/distancia)
            dentro de tam_banda -> + sep_banda; fuera, - tam_dist por
            unidad de tam fuera de la banda.
          - guía perdido: - sin_vista por paso sin ver la pelota y
            penalización terminal si pasan sin_ver_fin pasos.
          - colisión: penalización terminal si el IR frontal supera
            ir_choque.
          - carril: no es medible con estos sensores; se favorece con el
            término de centrado (- centrado * |x|), que mantiene la
            pelota en el centro de la imagen y por tanto al seguidor
            orientado hacia el guía, también en las curvas.
          - regularización: giro y cambio de acción.
        """
        k = self.cfg['recompensa']
        lo, hi = self.cfg['tam_banda']
        t = {}
        if self._visible:
            dist = max(0.0, lo - self._tam, self._tam - hi)
            t['sep'] = k['sep_banda'] if dist == 0.0 else -k['tam_dist'] * dist
            t['centrado'] = -k['centrado'] * abs(self._x)
            t['sin_vista'] = 0.0
        else:
            t['sep'] = 0.0
            t['centrado'] = 0.0
            t['sin_vista'] = -k['sin_vista']
        t['carril'] = 0.0
        t['giro'] = -k['giro'] * abs(w) / self.cfg['w_max']
        t['cambio'] = -k['cambio_accion'] * float(
            np.linalg.norm(accion - self._accion_prev)) / 2.0
        t['terminal'] = -k['terminal'] if causa in (
            'colision', 'guia_perdido') else 0.0
        return float(sum(t.values())), t

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._detener()

        if self.cli_reset is not None:
            # La escena vuelve a su estado inicial (ambos robots donde los
            # coloca el escenario, el guía vuelve a circular). En
            # entrenamiento se recoloca al seguidor detrás del guía con
            # separación/lateral/orientación aleatorios; en evaluación
            # (modo_evaluacion=True) no se toca: el seguidor parte parado
            # de donde lo deja el escenario (anexo B.2).
            self._reiniciar_escena()
            time.sleep(0.5)
            if self.cfg['aleatorizar_inicio'] and not self.modo_eval:
                try:
                    self._colocar_inicio()
                except RuntimeError as e:
                    if self.modo == 'verdad':
                        raise
                    print('[reset] sin colocación aleatoria:', e)
                time.sleep(0.2)
            self._esperar_poses(requiere_guia=(self.modo == 'verdad'))
        else:
            # Entorno real: el reinicio lo hace una persona.
            input('Coloca el seguidor detrás del guía y pulsa Enter... ')

        self.pasos = 0
        self.ultima_accion = np.zeros(2, dtype=np.float32)
        self._accion_prev = np.zeros(2, dtype=np.float32)
        self._cmd = (0.0, 0.0)
        self._reset_percepcion()

        self._vaciar_cola()
        self._esperar_lectura_nueva()
        self._actualizar_percepcion()
        self._t_ultimo_paso = time.time()
        return self._observacion(), {}

    def step(self, accion):
        c = self.cfg
        accion = np.clip(np.asarray(accion, dtype=np.float32), -1.0, 1.0)
        self._accion_prev = self.ultima_accion
        self.ultima_accion = accion

        # 1. Reescalar a unidades del SI y enviar por cmd_vel.
        v = c['v_min'] + (accion[0] + 1.0) * 0.5 * (c['v_max'] - c['v_min'])
        w = float(accion[1]) * c['w_max']
        self._publicar_velocidad(v, w)

        # 2. Dejar actuar sin hacer spin (los mensajes se acumulan).
        time.sleep(c['espera_accion_s'])

        # 3-4. Vaciar la cola y esperar una lectura posterior a la acción.
        self._vaciar_cola()
        self._esperar_lectura_nueva()
        self._actualizar_percepcion()
        # Refrescar también las poses reales, que llegan por otros tópicos.
        self._vaciar_cola()

        self.pasos += 1
        ahora = time.time()
        dt = ahora - self._t_ultimo_paso
        self._t_ultimo_paso = ahora

        # 5. Evaluar con la verdad del terreno (si hay simulador).
        causa = None
        verdad = self._verdad() if self.modo == 'verdad' else None
        if verdad is not None:
            sep, lat, d = verdad
            causa = self._causa_fin(sep, lat, d)
            recompensa, terminos = self._recompensa(
                sep, lat, w, accion, causa)
        else:
            # Modo 'sensores' (y entorno real): todo sale de blob + IR.
            sep = lat = d = None
            if c['usar_ir'] and self._ir_front >= c['ir_choque']:
                causa = 'colision'
            elif ((self._visto and self._sin_ver >= c['sin_ver_fin'])
                  or (not self._visto and self.pasos >= c['pasos_gracia'])):
                causa = 'guia_perdido'
            recompensa, terminos = self._recompensa_sensores(
                w, accion, causa)
            # Solo para registro (no interviene en recompensa ni fin).
            if self._pose is not None and self._pose_guia is not None:
                d = math.hypot(self._pose['x'] - self._pose_guia['x'],
                               self._pose['z'] - self._pose_guia['z'])

        terminated = causa is not None
        truncated = bool((not terminated) and self.pasos >= c['pasos_max'])
        if truncated:
            causa = 'limite_tiempo'

        if terminated or truncated:
            self._detener()

        lo, hi = c['banda_sep']
        lim = (c['ancho_carril'] - c['ancho_robot']) / 2.0
        info = {
            'causa': causa,                     # None mientras continúa
            'completado': bool(causa == 'limite_tiempo'),
            'dt': dt,                           # periodo real del paso (s)
            'v': float(v), 'w': float(w),
            'visible': self._visible,
            'sin_ver': self._sin_ver,
            # --- verdad del terreno, SOLO para registro y métricas
            'separacion': sep,
            'lateral': lat,
            'dist_centros': d,
            'en_banda': None if sep is None else bool(lo <= sep <= hi),
            'en_carril': None if lat is None else bool(abs(lat) <= lim),
            'pose_seguidor': None if self._pose is None else
                (self._pose['x'], self._pose['z'], self._pose['yaw']),
            'pose_guia': None if self._pose_guia is None else
                (self._pose_guia['x'], self._pose_guia['z'],
                 self._pose_guia['yaw']),
            'recompensa_terminos': terminos,
        }

        if self.verbose:
            print('paso {:3d}  sep={}  lat={}  vis={}  v={:+.3f}  w={:+.3f}  '
                  'r={:+.3f}  {}'.format(
                      self.pasos,
                      'NA' if sep is None else '{:.3f}'.format(sep),
                      'NA' if lat is None else '{:+.3f}'.format(lat),
                      int(self._visible), v, w, recompensa, causa or ''))

        return self._observacion(), recompensa, terminated, truncated, info

    def close(self):
        try:
            self._detener()
        except Exception:
            pass
        try:
            self.nodo.destroy_node()
        except Exception:
            pass
        # No se llama a rclpy.shutdown(): otros entornos del mismo proceso
        # seguirían necesitando el contexto.


# =====================================================================
# Utilidades de medida
# =====================================================================

def grabar_linea_central(config=None, salida=None, paso_m=0.02,
                         v_aprox=0.08, w_aprox=0.0, t_aprox_max=20.0,
                         umbral_mov_m=0.03):
    """Graba la línea central del carril: la trayectoria del guía en una
    vuelta completa (anexo B.1).

    El guía solo arranca cuando el seguidor se acerca, así que la función
    primero hace avanzar al seguidor por cmd_vel (v_aprox m/s, w_aprox
    rad/s) hasta detectar que el guía se mueve (se desplaza más de
    umbral_mov_m de su posición inicial). Entonces detiene al seguidor y
    empieza a grabar. Si en t_aprox_max segundos el guía no se mueve, se
    aborta. Para otra maniobra (girar, partir de otro sitio) colocar antes
    al seguidor con cmd_vel / set_robot_location y ajustar v_aprox y
    w_aprox.

    Requiere el contenedor sim del guía lanzado. Termina al cerrar la
    vuelta (vuelve a <0.15 m del punto de partida tras recorrer >2 m) o
    con Ctrl-C. Guarda un JSON {'puntos_m': [[x, z], ...]} en metros.
    """
    cfg = cargar_config(config)
    salida = salida or cfg['linea_central']
    if not rclpy.ok():
        rclpy.init()
    nodo = Node('p1_grabar_linea')
    pub = nodo.create_publisher(
        Twist, cfg['ns_robot'] + '/base/cmd_vel', 10)

    def mandar(v, w):
        m = Twist()
        m.linear.x = float(v)
        m.angular.z = float(w)
        pub.publish(m)

    def parar():
        for _ in range(3):
            mandar(0.0, 0.0)
            time.sleep(0.05)

    pts = []
    est = {'inicio': None, 'actual': None}

    def cb(m):
        x, z = m.position.x / 1000.0, m.position.z / 1000.0
        est['actual'] = (x, z)
        if est['inicio'] is None:
            est['inicio'] = (x, z)
        if not pts or math.hypot(x - pts[-1][0], z - pts[-1][1]) >= paso_m:
            pts.append([x, z])

    nodo.create_subscription(
        RobotLocation, cfg['ns_guia'] + '/sim/robot_location', cb, 1)

    def guia_movido():
        if est['inicio'] is None or est['actual'] is None:
            return False
        return math.hypot(est['actual'][0] - est['inicio'][0],
                          est['actual'][1] - est['inicio'][1]) > umbral_mov_m

    recorrido = 0.0
    try:
        # ---- Fase 1: acercar el seguidor hasta que el guía arranque.
        t0 = time.time()
        while rclpy.ok() and est['inicio'] is None:
            rclpy.spin_once(nodo, timeout_sec=0.1)
            if time.time() - t0 > 5.0:
                print('No llega la pose del guía ({}). ¿Está lanzado su '
                      'contenedor sim?'.format(cfg['ns_guia']))
                return None
        print('Acercando el seguidor hasta que el guía arranque...')
        t0 = time.time()
        while rclpy.ok() and not guia_movido():
            mandar(v_aprox, w_aprox)
            rclpy.spin_once(nodo, timeout_sec=0.1)
            if time.time() - t0 > t_aprox_max:
                parar()
                print('El guía no se ha movido en {:.0f} s. Ajustar la '
                      'maniobra (v_aprox, w_aprox) o acercar el seguidor '
                      'a mano.'.format(t_aprox_max))
                return None
        parar()
        print('El guía se mueve; seguidor detenido. Grabando la '
              'trayectoria ({})... Ctrl-C para terminar.'
              .format(cfg['ns_guia']))

        # ---- Fase 2: grabar una vuelta completa.
        while rclpy.ok():
            rclpy.spin_once(nodo, timeout_sec=0.1)
            if len(pts) > 1:
                recorrido = sum(math.hypot(pts[i][0] - pts[i - 1][0],
                                           pts[i][1] - pts[i - 1][1])
                                for i in range(1, len(pts)))
                if recorrido > 2.0 and math.hypot(
                        pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]) < 0.15:
                    print('Vuelta cerrada.')
                    break
    except KeyboardInterrupt:
        pass
    finally:
        try:
            parar()
        except Exception:
            pass
        nodo.destroy_node()
    if len(pts) < 10:
        print('Muy pocos puntos ({}); no se guarda.'.format(len(pts)))
        return None
    with open(salida, 'w') as f:
        json.dump({'puntos_m': pts}, f)
    print('{} puntos, {:.2f} m de recorrido -> {}'.format(
        len(pts), recorrido, salida))
    return salida


def calibrar(config=None, salida_csv='calibracion_sensor.csv'):
    """Caracterización del sensor (apartado 5.1) y medida de velocidades.

    Imprime y registra, de forma continua, lo que ve el seguidor (blob
    verde: x, y, size; IR frontal) y, si el módulo sim está disponible,
    la verdad del terreno: separación, error lateral y velocidad del guía.
    Escribe un CSV con una fila por mensaje de blobs. Al terminar
    (Ctrl-C) imprime un resumen por tramos de separación de 0.1 m:
    tasa de detección y media/desviación de x, y y size, que sirven para
    fijar blob_x_max, blob_y_max, blob_size_max, blob_size_min, etc.

    Procedimientos sugeridos:
      * Con el seguidor parado y el guía circulando: variación de size y
        posición con la distancia, ruido, rectas frente a curvas.
      * Con el seguidor en movimiento manual: efecto de la inclinación
        de la cámara (mover el tilt con base/move_tilt antes).
      * Repetir sobre el robot real con usar el mismo código (sin sim:
        solo se registra lo que ve el robot) y comparar los resúmenes.
      * Velocidad del guía (m/s) para cada configuración del anexo B:
        sirve para fijar v_max/v_min.
    """
    cfg = cargar_config(config)
    if not rclpy.ok():
        rclpy.init()
    nodo = Node('p1_calibracion')

    est = {'blobs': None, 'ir': None, 'f': None, 'g': None,
           'g_prev': None, 'v_g': float('nan')}
    nodo.create_subscription(
        BlobArray, cfg['ns_robot'] + '/smartphone/color_blobs',
        lambda m: est.__setitem__('blobs', m), 1)
    nodo.create_subscription(
        Int32MultiArray, cfg['ns_robot'] + '/base/ir',
        lambda m: est.__setitem__('ir', list(m.data)), 1)
    nodo.create_subscription(
        RobotLocation, cfg['ns_robot'] + '/sim/robot_location',
        lambda m: est.__setitem__('f', _pose_de_msg(m)), 1)

    def cb_g(m):
        p = _pose_de_msg(m)
        if est['g'] is not None and p['t'] > est['g']['t']:
            dt = p['t'] - est['g']['t']
            d = math.hypot(p['x'] - est['g']['x'], p['z'] - est['g']['z'])
            # media móvil exponencial de la velocidad del guía
            v = d / dt
            est['v_g'] = v if math.isnan(est['v_g']) else 0.9 * est['v_g'] + 0.1 * v
        est['g'] = p
    nodo.create_subscription(
        RobotLocation, cfg['ns_guia'] + '/sim/robot_location', cb_g, 1)

    linea = None
    if os.path.exists(cfg['linea_central']):
        linea = LineaCentral.desde_archivo(cfg['linea_central'])

    cab = ['t', 'visible', 'x', 'y', 'size', 'ir_front',
           'sep', 'lat', 'dist_centros', 'v_guia']
    filas = []
    print('{:>7s} {:>4s} {:>7s} {:>7s} {:>8s} {:>6s} {:>7s} {:>7s} {:>7s}'
          .format('t', 'vis', 'x', 'y', 'size', 'IR', 'sep', 'lat', 'v_guia'))

    t_ini = time.time()
    n_prev = -1
    try:
        while rclpy.ok():
            rclpy.spin_once(nodo, timeout_sec=0.5)
            m = est['blobs']
            if m is None or id(m) == n_prev:
                continue
            n_prev = id(m)
            b = extraer_blob(m, cfg['color_blob'])
            ir = (max(max(v, 0) for v in est['ir'][:5])
                  if est['ir'] else float('nan'))
            sep = lat = d = float('nan')
            if est['f'] and est['g']:
                f, g = est['f'], est['g']
                d = math.hypot(f['x'] - g['x'], f['z'] - g['z'])
                if linea is not None:
                    s_f, lat, _ = linea.proyectar(f['x'], f['z'])
                    s_g, _, _ = linea.proyectar(g['x'], g['z'])
                    sep = linea.separacion(s_g, s_f)
            fila = [time.time() - t_ini, int(b is not None),
                    b[0] if b else float('nan'),
                    b[1] if b else float('nan'),
                    b[2] if b else float('nan'),
                    ir, sep, lat, d, est['v_g']]
            filas.append(fila)
            print('{:7.1f} {:4d} {:7.1f} {:7.1f} {:8.1f} {:6.0f} {:7.3f} '
                  '{:+7.3f} {:7.3f}'.format(
                      fila[0], fila[1], fila[2], fila[3], fila[4], fila[5],
                      sep, lat, est['v_g']))
    except KeyboardInterrupt:
        pass
    finally:
        nodo.destroy_node()

    if not filas:
        print('Sin datos.')
        return
    with open(salida_csv, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(cab)
        wr.writerows(filas)
    print('\n{} filas -> {}'.format(len(filas), salida_csv))

    A = np.array(filas, dtype=float)
    if not np.all(np.isnan(A[:, 6])):
        print('\nResumen por separación (tramos de 0.1 m):')
        print('{:>10s} {:>4s} {:>7s} {:>16s} {:>16s} {:>18s}'.format(
            'sep (m)', 'n', 'detec', 'x media±std', 'y media±std',
            'size media±std'))
        for lo in np.arange(0.0, 1.5, 0.1):
            sel = (A[:, 6] >= lo) & (A[:, 6] < lo + 0.1)
            if not sel.any():
                continue
            vis = A[sel, 1] == 1
            def ms(col):
                v = A[sel, col][vis]
                return ('{:7.1f}±{:5.1f}'.format(v.mean(), v.std())
                        if len(v) else '       -      ')
            print('{:4.1f}-{:4.1f} {:4d} {:6.0%} {:>16s} {:>16s} {:>18s}'
                  .format(lo, lo + 0.1, int(sel.sum()), vis.mean(),
                          ms(2), ms(3), ms(4)))
        vg = A[:, 9][~np.isnan(A[:, 9])]
        if len(vg):
            print('\nVelocidad del guía: media {:.3f} m/s, máx {:.3f} m/s'
                  .format(vg.mean(), vg.max()))
    vis = A[:, 1] == 1
    if vis.any():
        print('\nRango observado: x [{:.0f}, {:.0f}]  y [{:.0f}, {:.0f}]  '
              'size [{:.0f}, {:.0f}]'.format(
                  np.nanmin(A[vis, 2]), np.nanmax(A[vis, 2]),
                  np.nanmin(A[vis, 3]), np.nanmax(A[vis, 3]),
                  np.nanmin(A[vis, 4]), np.nanmax(A[vis, 4])))
        print('-> ajustar blob_x_max, blob_y_max, blob_size_max y '
              'blob_size_min en el archivo de configuración.')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--calibrar', action='store_true',
                    help='caracterización del sensor (Ctrl-C para terminar)')
    ap.add_argument('--linea', action='store_true',
                    help='grabar la línea central siguiendo al guía')
    ap.add_argument('--config', default='config_p1.json')
    ap.add_argument('--escribir-config', action='store_true',
                    help='escribe el archivo de configuración por defecto')
    a = ap.parse_args()
    if a.escribir_config:
        guardar_config(a.config)
        print('Escrito', a.config)
    elif a.linea:
        grabar_linea_central(a.config)
    elif a.calibrar:
        calibrar(a.config)
    else:
        ap.print_help()