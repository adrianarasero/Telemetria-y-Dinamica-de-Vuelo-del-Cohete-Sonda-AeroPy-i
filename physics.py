import numpy as np
import math as m  # Mapea nuestro math.py modificado

# Datos de partida del AeroPy
voltaje_sensor = np.array([0.0, 1.25, 2.45, 3.80, 5.10])
fuerza_real    = np.array([0.0, 15.2, 31.0, 48.5, 65.3])

matriz_estatica = np.array([
    [1.0,  1.0,  1.0],  
    [0.0,  1.5, -1.5],  
    [-2.0, 1.0,  1.0]   
])
vector_cargas = np.array([25000.0, 1500.0, -800.0])  

h_atm = np.array([0, 2000, 4000, 6000, 8000, 10000, 12000, 14000])
T_atm = np.array([288.15, 275.15, 262.15, 249.15, 236.15, 223.15, 216.65, 216.65])
P_atm = np.array([101325, 79501, 61660, 47217, 35651, 26499, 19399, 14170])

t_vuelo = np.linspace(0, 60, 1200)  # Muestreo a 20 Hz[cite: 3]
v_vuelo = 800 * np.sin(np.pi * t_vuelo / 60) * np.exp(-0.02 * t_vuelo)[cite: 3]


def apartado_1():
    # Comparamos Gauss propio y LU
    reacciones_gauss = m.my_gauss(matriz_estatica, vector_cargas)
    reacciones_lu = m.solver_lu(matriz_estatica, vector_cargas)
    print(f"[Apartado 1] Reacciones (Gauss): R1={reacciones_gauss[0]:.2f}N, R2={reacciones_gauss[1]:.2f}N, R3={reacciones_gauss[2]:.2f}N")
    return reacciones_gauss


def apartado_2():
    # Comparamos regresión propia y numpy
    w_propio = m.my_regression(voltaje_sensor, fuerza_real, degree=1)
    m_np, c_np = m.numpy_regression(voltaje_sensor, fuerza_real, degree=1)
    # w_propio devuelve [c, m] para orden ascendente
    c, slope = w_propio[0], w_propio[1]
    print(f"[Apartado 2] Calibración Motor: F(V) = {slope:.4f} * V + {c:.4f}")
    return slope, c


def apartado_3():
    h_t = 300 * t_vuelo
    T_t = np.interp(h_t, h_atm, T_atm)
    print(f"[Apartado 3] Temperatura media estimada: {np.mean(T_t):.2f} K")
    return h_t, T_t


def apartado_5():
    dt = t_vuelo[1] - t_vuelo[0]
    a_long_centrados = m.my_3ptos_centrados(v_vuelo, dt)
    a_long_np = m.numpy_derivation(v_vuelo, t_vuelo)
    print(f"[Apartado 5] Aceleración longitudinal máxima: {np.max(a_long_centrados):.2f} m/s²")
    return t_vuelo, a_long_centrados


def apartado_6():
    dt = t_vuelo[1] - t_vuelo[0]
    h_trapecio = m.my_trapecio(v_vuelo, dt)
    h_simpson = m.my_simpson(v_vuelo, dt)
    print(f"[Apartado 6] Altitud al apogeo (Trapecio): {h_trapecio:.2f} m | (Simpson): {h_simpson:.2f} m")
    return h_trapecio
