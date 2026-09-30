import os
import random
import time
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import vizdoom as vzd

# ==========================================
# 1. DISPOSITIVO Y GPU
# ==========================================

if torch.cuda.is_available():
    device = torch.device("cuda")
    dev_name = torch.cuda.get_device_name(0)
else:
    device = torch.device("cpu")
    dev_name = "CPU"
print(f"--> Dispositivo activo (Baseline DQN): {device} ({dev_name})")

# ==========================================
# 2. RED NEURONAL CONVOLUCIONAL (CNN DQN)
# ==========================================
class DoomDQN(nn.Module):
    def __init__(self, n_actions):
        super(DoomDQN, self).__init__()
        # Entrada: [Batch, 1, 16, 32]
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        
        self.fc1 = nn.Linear(32 * 16 * 32, 128)
        self.fc2 = nn.Linear(128, n_actions)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Aplanar
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# Acciones discretas simplificadas para el DQN
ACTIONS = [
    [0, 0, 0, 0, 0, 0, 0],  # 0: Espera
    [1, 0, 0, 0, 0, 0, 0],  # 1: Izquierda
    [0, 1, 0, 0, 0, 0, 0],  # 2: Derecha
    [0, 0, 1, 0, 0, 0, 0],  # 3: Fuego
    [0, 0, 0, 1, 0, 0, 0],  # 4: Strafe Izq
    [0, 0, 0, 0, 1, 0, 0]   # 5: Strafe Der
]
n_actions = len(ACTIONS)

policy_net = DoomDQN(n_actions).to(device)
target_net = DoomDQN(n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
memory = deque(maxlen=10000)

# CARGA DE PESOS PERSISTENTES (SI EXISTEN)
WEIGHTS_SAVE_FILE = "data/dqn_pesos.pth"
if os.path.exists(WEIGHTS_SAVE_FILE):
    policy_net.load_state_dict(torch.load(WEIGHTS_SAVE_FILE, map_location=device, weights_only=True))
    target_net.load_state_dict(policy_net.state_dict())
    print(f"--> Pesos de DQN cargados correctamente desde: {WEIGHTS_SAVE_FILE}")
else:
    print("--> No se encontraron pesos previos de DQN. Iniciando desde cero.")

# Parámetros de RL
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.995
epsilon = EPSILON_START

# ==========================================
# 3. CONFIGURACIÓN VIZDOOM
# ==========================================
game = vzd.DoomGame()
scenarios_dir = vzd.scenarios_path
game.load_config(os.path.join(scenarios_dir, "defend_the_center.cfg"))

game.add_available_button(vzd.Button.MOVE_LEFT)
game.add_available_button(vzd.Button.MOVE_RIGHT)
game.add_available_button(vzd.Button.MOVE_BACKWARD)
game.add_available_button(vzd.Button.MOVE_FORWARD)

game.set_window_visible(True)
game.set_mode(vzd.Mode.PLAYER)
game.set_screen_format(vzd.ScreenFormat.GRAY8)
game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)

game.add_available_game_variable(vzd.GameVariable.HEALTH)
game.add_available_game_variable(vzd.GameVariable.AMMO2)
game.add_available_game_variable(vzd.GameVariable.KILLCOUNT)
game.init()

print("\n==========================================================================")
print("  BASELINE TRADICIONAL: DEEP Q-NETWORK (DQN + CNN) [ACTUALIZADO]")
print("==========================================================================\n")

FRAME_REPEAT = 2
LOTE_EPISODIOS = 50
ep = 0

try:
    while True:
        print(f"\n--- INICIANDO LOTE DQN: Episodios {ep+1} al {ep+LOTE_EPISODIOS} ---")
        
        for _ in range(LOTE_EPISODIOS):
            ep += 1
            game.new_episode()
            step = 0
            
            init_state = game.get_state()
            last_health = init_state.game_variables[0] if (init_state and len(init_state.game_variables) > 0) else 100.0
            last_kills  = init_state.game_variables[2] if (init_state and len(init_state.game_variables) > 2) else 0.0

            while not game.is_episode_finished():
                state = game.get_state()
                screen = state.screen_buffer
                current_health = state.game_variables[0] if len(state.game_variables) > 0 else 100.0
                current_kills  = state.game_variables[2] if len(state.game_variables) > 2 else last_kills

                # Procesamiento visual (16x32)
                screen_t = torch.from_numpy(screen).float().to(device).unsqueeze(0).unsqueeze(0)
                retina = F.interpolate(screen_t, size=(16, 32), mode="area")
                retina_norm = (retina - retina.mean()) / (retina.std() + 1e-5)

                # Selección de acción (Epsilon-greedy)
                if random.random() < epsilon:
                    action_idx = random.randint(0, n_actions - 1)
                else:
                    with torch.no_grad():
                        q_values = policy_net(retina_norm)
                        action_idx = q_values.max(1)[1].item()

                action = ACTIONS[action_idx]
                reward_env = game.make_action(action, FRAME_REPEAT)
                step += 1

                # Recompensa compuesta con castigo por inactividad
                health_delta = current_health - last_health
                kill_confirmed = (current_kills - last_kills) > 0
                
                custom_reward = reward_env
                if kill_confirmed:
                    custom_reward += 5.0
                if health_delta < 0:
                    custom_reward += health_delta * 0.2
                
                # Castigo leve si elige la acción 0 (espera/no hacer nada) para romper bloqueos
                if action_idx == 0:
                    custom_reward -= 0.05

                # Siguiente estado
                next_state_obs = game.get_state()
                if next_state_obs is not None:
                    next_screen = next_state_obs.screen_buffer
                    next_screen_t = torch.from_numpy(next_screen).float().to(device).unsqueeze(0).unsqueeze(0)
                    next_retina = F.interpolate(next_screen_t, size=(16, 32), mode="area")
                    next_retina_norm = (next_retina - next_retina.mean()) / (next_retina.std() + 1e-5)
                else:
                    next_retina_norm = torch.zeros_like(retina_norm)

                done = game.is_episode_finished()
                memory.append((retina_norm, action_idx, custom_reward, next_retina_norm, done))

                # Entrenamiento por lotes (Experience Replay)
                if len(memory) >= BATCH_SIZE:
                    batch = random.sample(memory, BATCH_SIZE)
                    state_batch = torch.cat([b[0] for b in batch])
                    action_batch = torch.tensor([b[1] for b in batch], device=device).unsqueeze(1)
                    reward_batch = torch.tensor([b[2] for b in batch], device=device, dtype=torch.float32)
                    next_state_batch = torch.cat([b[3] for b in batch])
                    done_batch = torch.tensor([b[4] for b in batch], device=device, dtype=torch.float32)

                    state_action_values = policy_net(state_batch).gather(1, action_batch)
                    next_state_values = target_net(next_state_batch).max(1)[0].detach()
                    expected_state_action_values = reward_batch + (GAMMA * next_state_values * (1 - done_batch))

                    loss = nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(1))
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                last_health = current_health
                last_kills = current_kills

                if step % 50 == 0:
                    print(f"DQN Ep {ep:03d} | Paso {step:03d} | Kills: {current_kills:2.0f} | Epsilon: {epsilon:.2f}")

            if epsilon > EPSILON_END:
                epsilon *= EPSILON_DECAY

            print(f"--> Fin Ep DQN {ep} | Kills: {last_kills}\n")

        # Respaldo automático de pesos
        torch.save(policy_net.state_dict(), WEIGHTS_SAVE_FILE)
        print(f"--> Lote de DQN completado con éxito. Pesos respaldados en: {WEIGHTS_SAVE_FILE}")

        continuar = input("¿Deseas ejecutar otro lote de DQN? (s/n): ")
        if continuar.lower() != 's':
            break

except KeyboardInterrupt:
    print("\n\n--> Interrupción manual del baseline DQN.")
finally:
    torch.save(policy_net.state_dict(), WEIGHTS_SAVE_FILE)
    print(f"--> Pesos de DQN guardados de forma segura en: {WEIGHTS_SAVE_FILE}")
    game.close()
    print("Baseline DQN finalizado.")
