import numpy as np
import matplotlib.pyplot as plt

# Parámetros del modelo Dynamic CC (Lung data, Tabla 3)
a = 0.399 # day^-1
b = 2.66 # mm^-2 * day^-1

# Condiciones iniciales
V0 = 1.0 # mm^3 (volumen tumoral inicial)
K0 = 2.6 # mm^3 (capacidad de carga inicial)
t0 = 0.0 # día
tf = 40.0 # tiempo final en días
h  = 0.1 # paso de integración en días

# Sistema de EDOs
def f(V, K):
    dVdt = a * V * np.log(K / V)
    dKdt = b * (V ** (2.0/3.0))
    return dVdt, dKdt

# Integrador de Runge–Kutta 4
def rk4(V0, K0, t0, tf, h):
    N = int((tf - t0) / h) + 1
    t = np.linspace(t0, tf, N)
    V = np.zeros(N)
    K = np.zeros(N)

    V[0], K[0] = V0, K0

    for i in range(N - 1):
        dV1, dK1 = f(V[i], K[i])
        dV2, dK2 = f(V[i] + 0.5*h*dV1, K[i] + 0.5*h*dK1)
        dV3, dK3 = f(V[i] + 0.5*h*dV2, K[i] + 0.5*h*dK2)
        dV4, dK4 = f(V[i] + h*dV3, K[i] + h*dK3)

        V[i+1] = V[i] + (h/6.0)*(dV1 + 2*dV2 + 2*dV3 + dV4)
        K[i+1] = K[i] + (h/6.0)*(dK1 + 2*dK2 + 2*dK3 + dK4)

    return t, V, K

# Correr simulación
t, V, K = rk4(V0, K0, t0, tf, h)

# Graficar resultados
plt.figure(figsize=(8,4))
plt.plot(t, V, label="Volumen tumoral V(t)")
plt.plot(t, K, label="Capacidad de carga K(t)")
plt.xlabel("Tiempo (días)")
plt.ylabel("Volumen (mm^3)")
plt.title("Modelo Dynamic CC (método numérico tradicional, RK4)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("dynamic_cc_model.png", dpi=300) 
print("Imagen guardada como 'dynamic_cc_model.png'")

