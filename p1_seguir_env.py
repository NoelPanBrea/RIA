#!/usr/bin/env python3
"""
Practica 1 - Entorno de Gymnasium para seguir al robot guia.

El seguidor ve la pelota verde del guia con el modulo de blobs y tiene
que mantenerla centrada y a buena distancia.

Lanzar antes el puente:
    ros2 run robobo_ros2 robobo_container --ros-args \
         -p ip:=host.docker.internal -p modules:="['blob','sim']"
"""

import math
import time

import numpy as np

import gymnasium as gym
from gymnasium import spaces

import rclpy
from rclpy.node import Node

from std_msgs.msg import Int32MultiArray
from geometry_msgs.msg import Twist

# Interfaces propias del puente. El módulo sim las usa para hablar con
# el simulador; no existen cuando se trabaja con el robot real.
from robobo_ros2_interfaces.msg import BlobArray, RobotLocation
from robobo_ros2_interfaces.srv import ResetSimulation


# =====================================================================
# Constantes del problema
# =====================================================================

# Espacio de nombres del nodo puente. Con los valores por defecto de
# robobo_container, robot_name vale "0".
NS = '/robobo/robot_0'
NS_BASE = NS + '/base'
NS_PHONE = NS + '/smartphone'

# Espacio de nombres del módulo sim, dentro del mismo nodo puente. Es el
# canal privilegiado: existe en el simulador y no en el robot real.
NS_SIM = NS + '/sim'

# ---------- Calibracion (medir con --calibrar) ----------

CALIBRADO = False   # poner a True cuando esten medidos

BLOB_X_MAX = 100.0  # la x del blob va de 0 a 100

# size del blob a 0.45 m, a 0.20 m y casi tocando al guia
SIZE_LEJOS = 400.0
SIZE_CERCA = 2000.0
SIZE_SAT = 5500.0

# Orden de los ocho valores del array publicado en <NS_BASE>/ir:
#   0 FrontLL  1 FrontL  2 FrontC  3 FrontR  4 FrontRR
#   5 BackL    6 BackC   7 BackR
IR_FRONTALES = [0, 1, 2, 3, 4]
IR_SATURACION = 1000.0
P_CHOQUE = 0.85

V_MAX = 0.15   # m/s, tiene que ser mas que el guia
W_MAX = 1.0    # rad/s

TILT_ANGULO = 90.0      # grados
TILT_VELOCIDAD = 15.0
ESPERA_INICIO_S = 2.0   # espera al empezar cada episodio


# ---------- Resto de constantes ----------

# uso la raiz del size porque va con 1/distancia
T_LEJOS = math.sqrt(SIZE_LEJOS / SIZE_SAT)
T_CERCA = math.sqrt(SIZE_CERCA / SIZE_SAT)
T_MUY_LEJOS = T_LEJOS * 0.45 / 0.90   # equivale a 0.90 m

PASO_S = 0.15        # espera despues de mandar la accion
ESPERA_MAX_S = 3.0
T_PERDIDO_S = 3.0    # segundos sin ver la pelota para darlo por perdido

# pesos de la recompensa
K_SEP = 1.0
K_CENTRO = 0.5
K_NO_VER = 0.5
K_GIRO = 0.05
R_CHOQUE = 10.0
R_PERDIDO = 10.0


def lista_blobs(msg):
    # cojo el primer campo del mensaje que sea una lista
    for nombre, tipo in msg.get_fields_and_field_types().items():
        if tipo.startswith('sequence') or tipo.endswith(']'):
            return getattr(msg, nombre)
    return []


def blob_verde(msg):
    """Devuelve (x, y, size) del blob verde o None si no se ve."""
    for b in lista_blobs(msg):
        if 'green' in str(b.color).lower() and b.size > 0:
            return (float(b.x), float(b.y), float(b.size))
    return None


def x_normalizada(x):
    # 0 en el centro, positivo a la derecha
    mitad = BLOB_X_MAX / 2.0
    return float(min(max((x - mitad) / mitad, -1.0), 1.0))


def tam_normalizado(size):
    return float(min(math.sqrt(max(size, 0.0) / SIZE_SAT), 1.0))


def proximidad_ir(ir):
    if ir is None:
        return 0.0
    return float(max(min(max(ir[i], 0) / IR_SATURACION, 1.0)
                     for i in IR_FRONTALES))


def activar_solo_verde(nodo):
    """Llama al servicio para que solo detecte el verde."""
    nombre = NS_PHONE + '/set_active_color_blobs'
    try:
        from rosidl_runtime_py.utilities import get_service

        tipos = None
        t0 = time.time()
        while tipos is None and time.time() - t0 < 3.0:
            tipos = dict(nodo.get_service_names_and_types()).get(nombre)
            if tipos is None:
                rclpy.spin_once(nodo, timeout_sec=0.1)
        if tipos is None:
            raise RuntimeError('el servicio no aparece')

        Servicio = get_service(tipos[0])
        cliente = nodo.create_client(Servicio, nombre)
        if not cliente.wait_for_service(timeout_sec=2.0):
            raise RuntimeError('el servicio no responde')

        peticion = Servicio.Request()
        for color in ('red', 'green', 'blue', 'custom'):
            setattr(peticion, color, color == 'green')
        futuro = cliente.call_async(peticion)
        rclpy.spin_until_future_complete(nodo, futuro, timeout_sec=3.0)
    except Exception as e:
        print('AVISO: no se ha podido activar solo el verde ({}).'.format(e))

def mover_tilt(nodo):
    """Pone el tilt en TILT_ANGULO con el servicio base/move_tilt."""
    nombre = NS_BASE + '/move_tilt'
    try:
        from rosidl_runtime_py.utilities import get_service

        tipos = None
        t0 = time.time()
        while tipos is None and time.time() - t0 < 3.0:
            tipos = dict(nodo.get_service_names_and_types()).get(nombre)
            if tipos is None:
                rclpy.spin_once(nodo, timeout_sec=0.1)
        if tipos is None:
            raise RuntimeError('el servicio no aparece')

        Servicio = get_service(tipos[0])
        cliente = nodo.create_client(Servicio, nombre)
        if not cliente.wait_for_service(timeout_sec=2.0):
            raise RuntimeError('el servicio no responde')

        peticion = Servicio.Request()
        peticion.angle = TILT_ANGULO
        peticion.speed = TILT_VELOCIDAD
        futuro = cliente.call_async(peticion)
        rclpy.spin_until_future_complete(nodo, futuro, timeout_sec=3.0)
        nodo.destroy_client(cliente)
    except Exception as e:
        print('AVISO: no se ha podido mover el tilt ({}).'.format(e))

class RoboboSeguirEnv(gym.Env):
    """Seguir al guia por la pelota verde.

    Observacion (7 valores):
        0  x del blob, de -1 a 1 (si no se ve, la ultima)
        1  tamano del blob, de 0 a 1 (si no se ve, el ultimo)
        2  1 si se ve la pelota, 0 si no
        3  tiempo sin verla, de 0 a 1
        4  proximidad de los IR frontales
        5  velocidad lineal del paso anterior
        6  velocidad angular del paso anterior

    Accion (2 valores de -1 a 1):
        0  velocidad lineal, de 0 a V_MAX (no hay marcha atras)
        1  velocidad angular, de -W_MAX a W_MAX

    El episodio termina si choca o si pierde al guia mas de 3 s.
    """

    metadata = {'render_modes': []}

    def __init__(self, pasos_max=300, usar_simulador=True, verbose=False):
        super().__init__()

        self.pasos_max = pasos_max
        self.verbose = verbose

        if not CALIBRADO:
            print('AVISO: faltan por calibrar las constantes de calibracion '
                  '(CALIBRADO = False).')

        if not rclpy.ok():
            rclpy.init()
        self.nodo = Node('p1_seguir_entorno')

        self.pub_vel = self.nodo.create_publisher(
            Twist, NS_BASE + '/cmd_vel', 10)

        self._n_msgs = 0
        self._n_blob = 0
        self._blob = None
        self._ir = None
        self.nodo.create_subscription(
            BlobArray, NS_PHONE + '/color_blobs', self._cb_blobs, 1)
        self.nodo.create_subscription(
            Int32MultiArray, NS_BASE + '/ir', self._cb_ir, 1)

        # el modulo sim solo se usa para reiniciar y guardar la pose
        self.cli_reset = None
        self._pose = None
        if usar_simulador:
            self.cli_reset = self.nodo.create_client(
                ResetSimulation, NS_SIM + '/reset_simulation')
            if not self.cli_reset.wait_for_service(timeout_sec=5.0):
                raise RuntimeError(
                    'No responde el servicio {}/reset_simulation.\n'
                    'Lanzar robobo_container con el modulo sim o usar '
                    '--sin-simulador.'.format(NS_SIM))
            self.nodo.create_subscription(
                RobotLocation, NS_SIM + '/robot_location', self._cb_pose, 1)

        activar_solo_verde(self.nodo)

        self.observation_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0],
                         dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                          dtype=np.float32),
            dtype=np.float32)

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.pasos = 0
        self.ultima_accion = np.array([-1.0, 0.0], dtype=np.float32)
        self._x_ult = 0.0
        self._tam_ult = 0.0
        self._t_visto = time.time()

        # esperar a que llegue algo antes de empezar
        self._esperar_blob_nuevo(inicial=True)
        t0 = time.time()
        while self._ir is None and time.time() - t0 < 2.0:
            rclpy.spin_once(self.nodo, timeout_sec=0.05)
        if self._ir is None:
            print('AVISO: no llegan los IR, no se va a detectar el choque.')

    # ---------- ROS ----------

    def _cb_blobs(self, msg):
        self._blob = blob_verde(msg)
        self._n_blob += 1
        self._n_msgs += 1

    def _cb_ir(self, msg):
        self._ir = list(msg.data)
        self._n_msgs += 1

    def _cb_pose(self, msg):
        self._pose = (msg.position.x, msg.position.z, msg.rotation.y)
        self._n_msgs += 1

    def _vaciar_cola(self):
        # tirar los mensajes viejos (hasta 3 spin seguidos sin nada)
        vacias = 0
        for _ in range(100):
            antes = self._n_msgs
            rclpy.spin_once(self.nodo, timeout_sec=0.0)
            vacias = vacias + 1 if self._n_msgs == antes else 0
            if vacias >= 3:
                return

    def _esperar_blob_nuevo(self, inicial=False):
        # espera un mensaje de blobs nuevo, asi la observacion es
        # posterior a la accion
        objetivo = self._n_blob + 1
        t0 = time.time()
        while self._n_blob < objetivo:
            rclpy.spin_once(self.nodo, timeout_sec=0.05)
            if time.time() - t0 > ESPERA_MAX_S:
                raise RuntimeError(
                    'No llegan mensajes de {}/color_blobs. Mirar que el '
                    'puente este lanzado con el modulo blob.'.format(NS_PHONE)
                    if inicial else
                    'Han dejado de llegar los blobs en mitad del episodio.')

    def _publicar_velocidad(self, v, w):
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.pub_vel.publish(msg)

    def _detener(self):
        for _ in range(3):
            self._publicar_velocidad(0.0, 0.0)
            time.sleep(0.05)

    def _reiniciar_escena(self):
        futuro = self.cli_reset.call_async(ResetSimulation.Request())
        rclpy.spin_until_future_complete(self.nodo, futuro, timeout_sec=5.0)
        if futuro.result() is None or not futuro.result().success:
            raise RuntimeError('Ha fallado el reinicio de la escena.')

    # ---------- Observacion ----------

    def _percibir(self):
        """Devuelve (visible, segundos sin ver la pelota, proximidad IR)."""
        ahora = time.time()
        visible = self._blob is not None
        if visible:
            self._x_ult = x_normalizada(self._blob[0])
            self._tam_ult = tam_normalizado(self._blob[2])
            self._t_visto = ahora
        return visible, ahora - self._t_visto, proximidad_ir(self._ir)

    def _observacion(self, visible, sin_ver, prox):
        return np.array([
            self._x_ult,
            self._tam_ult,
            1.0 if visible else 0.0,
            min(sin_ver / T_PERDIDO_S, 1.0),
            prox,
            self.ultima_accion[0],
            self.ultima_accion[1],
        ], dtype=np.float32)

    # ---------- Recompensa ----------

    def _recompensa(self, visible, x, tam, giro, choque, perdido):
        if visible:
            # separacion: 1 dentro de la banda, baja a 0 fuera
            if tam < T_LEJOS:
                r_sep = max((tam - T_MUY_LEJOS) / (T_LEJOS - T_MUY_LEJOS),
                            0.0)
            elif tam > T_CERCA:
                r_sep = max((1.0 - tam) / (1.0 - T_CERCA), 0.0)
            else:
                r_sep = 1.0
            # mas premio cuanto mas centrada
            r = K_SEP * r_sep + K_CENTRO * (1.0 - abs(x))
        else:
            r = -K_NO_VER

        r -= K_GIRO * abs(giro)

        if choque:
            r -= R_CHOQUE
        if perdido:
            r -= R_PERDIDO

        return float(r)

    # ---------- Gymnasium ----------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._detener()

        if self.cli_reset is not None:
            self._reiniciar_escena()
            time.sleep(1.0)
        else:
            input('\nColoca el seguidor detras del guia y pulsa Intro... ')
        
        mover_tilt(self.nodo)
        time.sleep(ESPERA_INICIO_S)
        self.pasos = 0
        self.ultima_accion = np.array([-1.0, 0.0], dtype=np.float32)  # parado
        self._x_ult = 0.0
        self._tam_ult = 0.0

        self._vaciar_cola()
        self._esperar_blob_nuevo()

        self._t_visto = time.time()
        visible, sin_ver, prox = self._percibir()
        return self._observacion(visible, sin_ver, prox), {}

    def step(self, accion):
        accion = np.clip(np.asarray(accion, dtype=np.float32), -1.0, 1.0)

        v = (float(accion[0]) + 1.0) * 0.5 * V_MAX
        w = float(accion[1]) * W_MAX
        self._publicar_velocidad(v, w)

        time.sleep(PASO_S)

        self._vaciar_cola()
        self._esperar_blob_nuevo()

        self.pasos += 1
        self.ultima_accion = accion
        visible, sin_ver, prox = self._percibir()

        choque = prox >= P_CHOQUE
        perdido = sin_ver > T_PERDIDO_S
        en_banda = visible and T_LEJOS <= self._tam_ult <= T_CERCA

        recompensa = self._recompensa(
            visible, self._x_ult, self._tam_ult, float(accion[1]),
            choque, perdido)

        terminated = bool(choque or perdido)
        truncated = bool(self.pasos >= self.pasos_max)

        if terminated or truncated:
            self._detener()

        info = {
            'visible': visible,
            'en_banda': en_banda,
            'x': self._x_ult,
            'tam': self._tam_ult,
            'size': self._blob[2] if visible else 0.0,
            'proximidad': prox,
            'choque': choque,
            'perdido': perdido,
            'pose_sim': self._pose,
        }

        if self.verbose:
            print('paso {:3d}  {}  x={:+.2f}  tam={:.2f}  ir={:.2f}  '
                  'v={:.2f}  w={:+.2f}  r={:+.2f}{}'
                  .format(self.pasos, 'VE' if visible else '--',
                          self._x_ult, self._tam_ult, prox, v, w, recompensa,
                          '  CHOQUE' if choque else
                          ('  PERDIDO' if perdido else '')))

        return (self._observacion(visible, sin_ver, prox), recompensa,
                terminated, truncated, info)

    def close(self):
        try:
            self._detener()
        except Exception:
            pass
        try:
            self.nodo.destroy_node()
        except Exception:
            pass


def calibrar():
    """Imprime el blob verde y los IR para calibrar. Se para con Ctrl-C."""
    if not rclpy.ok():
        rclpy.init()
    nodo = Node('p1_seguir_medida')

    estado = {'msg': None, 'n': 0, 'ir': None}

    def cb_blobs(m):
        estado['msg'] = m
        estado['n'] += 1

    nodo.create_subscription(
        BlobArray, NS_PHONE + '/color_blobs', cb_blobs, 1)
    nodo.create_subscription(
        Int32MultiArray, NS_BASE + '/ir',
        lambda m: estado.__setitem__('ir', list(m.data)), 1)

    activar_solo_verde(nodo)
    mover_tilt(nodo)
    
    print('{:>8s} {:>8s} {:>9s} {:>8s} {:>8s} {:>8s}'
          .format('x', 'y', 'size', 'x_norm', 'tam', 'ir_max'))

    vistos = 0
    crudo_impreso = False
    t_ultimo = time.time()
    try:
        while rclpy.ok():
            rclpy.spin_once(nodo, timeout_sec=0.5)
            if estado['n'] == vistos:
                if time.time() - t_ultimo > 3.0:
                    print('No llegan mensajes de {}/color_blobs'
                          .format(NS_PHONE))
                    t_ultimo = time.time()
                continue
            vistos = estado['n']
            t_ultimo = time.time()

            msg = estado['msg']
            if not crudo_impreso and len(lista_blobs(msg)) > 0:
                # el primer mensaje entero, para ver como viene
                print('Mensaje:', msg)
                crudo_impreso = True

            ir = estado['ir']
            ir_max = max(ir[i] for i in IR_FRONTALES) if ir else 0
            verde = blob_verde(msg)
            if verde is None:
                print('{:>44s} {:8d}'.format('(no se ve la pelota)', ir_max))
            else:
                x, y, size = verde
                print('{:8.1f} {:8.1f} {:9.1f} {:+8.2f} {:8.2f} {:8d}'
                      .format(x, y, size, x_normalizada(x),
                              tam_normalizado(size), ir_max))
    except KeyboardInterrupt:
        pass
    finally:
        nodo.destroy_node()