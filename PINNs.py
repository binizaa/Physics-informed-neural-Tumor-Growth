import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. CARGA Y PROCESAMIENTO DE DATOS

df = pd.read_csv('LLC_sc_CCSB.txt')

t_data = df['Time'].values
V_data = df['Vol'].values

# --- NORMALIZACIÓN  ---
t_max = np.max(t_data)
V_max = np.max(V_data)

t_norm = t_data / t_max
V_norm = V_data / V_max

t_tensor = torch.tensor(t_norm, dtype=torch.float32).view(-1, 1).requires_grad_(True)
V_tensor = torch.tensor(V_norm, dtype=torch.float32).view(-1, 1)

print(f"Datos cargados: {len(df)} registros.")
print(f"Factores de escala -> Tiempo máx: {t_max}, Volumen máx: {V_max}")

# 2. DEFINICIÓN DE LA PINN
class TumorPINN(nn.Module):
    def __init__(self):
        super(TumorPINN, self).__init__()
        
        # Red Neuronal: Entrada t -> Salida (V_hat, K_hat)
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 2) 
        )
        
        # Parámetros biológicos entrenables 
        self.raw_a = nn.Parameter(torch.tensor([0.5]))
        self.raw_b = nn.Parameter(torch.tensor([0.5]))
        self.raw_d = nn.Parameter(torch.tensor([0.1]))

    def forward(self, t):
        return self.net(t)

    def get_params(self):
        a = torch.nn.functional.softplus(self.raw_a)
        b = torch.nn.functional.softplus(self.raw_b)
        d = torch.nn.functional.softplus(self.raw_d)
        return a, b, d

    def physics_loss(self, t, V_obs):
        # 1. Predicciones
        predictions = self.forward(t)
        V_hat = predictions[:, 0:1] 
        K_hat = predictions[:, 1:2] 
        
        # Recuperar escalas para las Ecuaciones Diferenciales
        # V_real = V_hat * V_max
        # t_real = t * t_max
        
        dV_dt_norm = torch.autograd.grad(V_hat, t, torch.ones_like(V_hat), create_graph=True)[0]
        dK_dt_norm = torch.autograd.grad(K_hat, t, torch.ones_like(K_hat), create_graph=True)[0]
        
        dV_dt = dV_dt_norm / t_max 
        dK_dt = dK_dt_norm / t_max
        
        a, b, d = self.get_params()
        
        # 3. Física (EDOs)
        # Ecuación 1: dV/dt = a * V * log(K/V)
        res_V = dV_dt - a * V_hat * torch.log(torch.abs(K_hat/V_hat) + 1e-6)
        
        # Ecuación 2: dK/dt = b * V^(2/3) - d * K
        term_b = b * (V_hat**(2/3)) * (V_max**(2/3) / V_max) # Ajuste de escala
        term_d = d * K_hat
        
        res_K = dK_dt - (term_b - term_d)

        # 4. Pérdidas
        loss_data = torch.mean((V_hat - V_obs)**2)  
        loss_physics = torch.mean(res_V**2) + torch.mean(res_K**2) 
        
        # Forzar condición inicial K0 approx V0 
        mask_t0 = (t < 0.1) 
        if mask_t0.any():
            loss_init = torch.mean((K_hat[mask_t0] - V_hat[mask_t0])**2)
        else:
            loss_init = 0.0

        return loss_data * 10 + loss_physics + loss_init

# 3. ENTRENAMIENTO
model = TumorPINN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

loss_history = []
print("Entrenando red neuronal con datos reales...")

for epoch in range(10001):
    optimizer.zero_grad()
    
    loss = model.physics_loss(t_tensor, V_tensor)
    
    loss.backward()
    optimizer.step()
    
    loss_history.append(loss.item())
    
    if epoch % 1000 == 0:
        a, b, d = model.get_params()
        print(f"Epoch {epoch}: Loss {loss.item():.5f} | a={a.item():.3f}, b={b.item():.3f}, d={d.item():.3f}")

# 4. RESULTADOS Y VISUALIZACIÓN
t_plot = torch.linspace(0, 1, 100).view(-1, 1)
with torch.no_grad():
    preds = model(t_plot)
    V_pred_norm = preds[:, 0].numpy()
    K_pred_norm = preds[:, 1].numpy()

t_plot_real = t_plot.numpy() * t_max
V_pred_real = V_pred_norm * V_max
K_pred_real = K_pred_norm * V_max

# Imprimir parámetros finales
a_final, b_final, d_final = model.get_params()
print("\n=== PARÁMETROS BIOLÓGICOS ESTIMADOS ===")
print(f"Tasa de crecimiento (a): {a_final.item():.4f}")
print(f"Estimulación angiogénica (b): {b_final.item():.4f}")
print(f"Inhibición vascular (d): {d_final.item():.4f}")

# Gráfica
plt.figure(figsize=(10, 6))
plt.scatter(df['Time'], df['Vol'], c='red', alpha=0.3, label='Datos Reales (Todos los IDs)')
plt.plot(t_plot_real, V_pred_real, 'b-', linewidth=3, label='PINN: Volumen Tumoral (V)')
plt.plot(t_plot_real, K_pred_real, 'g--', linewidth=3, label='PINN: Capacidad de Carga (K)')

plt.title('Modelo de Capacidad de Carga Dinámica (PINN Population Fit)')
plt.xlabel('Tiempo (Días)')
plt.ylabel('Volumen ($mm^3$)')
plt.legend()
plt.grid(True)
plt.savefig('resultado_reto.png')
plt.show()

