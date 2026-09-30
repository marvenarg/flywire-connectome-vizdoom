import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import vizdoom as vzd

# ==========================================
# 1. DISPOSITIVO Y GPU
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--> Dispositivo activo (Modo Evaluación DQN): {device} ({torch.cuda.get_device_name(0)})")

# ==========================================
# 2. RED NEURONAL CONVOLUCIONAL (CNN DQN)
# ==========================================
class DoomDQN(nn.Module):
    def __init__(self, n_actions):
        super(DoomDQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(32 * 16 * 32, 128)
        self.fc2 = nn.Linear(128, n_actions)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# Mismas acciones discretas
ACTIONS = [
    [0, 0, 0, 0, 0, 0, 0],  # Espera
    [1, 0, 0, 0, 0, 0, 0],  # Izquierda
    [0, 1, 0, 0, 0, 0, 0],  # Derecha
    [0, 0, 1, 0, 0, 0, 0],  # Fuego
    [0, 0, 0, 1, 0, 0, 0],  # Strafe Izq
    [0, 0, 0, 0, 1, 0, 0]   # Strafe Der
]
n_actions = len(ACTIONS)

policy_net = DoomDQN(n_actions).to(device)

# CARGA DE PESOS ENTRENADOS (CONGELADOS)
WEIGHTS_SAVE_FILE = "dqn_pesos.pth"
if os.path.exists(WEIGHTS_SAVE_FILE):
    policy_net.load_state_dict(torch.load(WEIGHTS_SAVE_FILE, map_location=device))
    policy_net.eval()  # Modo evaluación (congela capas si tuviera Dropout/Batchnorm)
    print(f"--> Pesos de DQN cargados correctamente para evaluación desde: {WEIGHTS_SAVE_FILE}")
else:
    raise FileNotFoundError(f"No se encontró el archivo '{WEIGHTS_SAVE_FILE}'. ¡Debes entrenarlo primero!")

# ==========================================
# 3. CONFIGURACIÓN VIZDOOM (MODO VISIBLE)
# ==========================================
game = vzd.DoomGame()
scenarios_dir = vzd.scenarios_path
game.load_config(os.path.join(scenarios_dir, "defend_the_center.cfg"))

game.add_available_button(vzd.Button.MOVE_LEFT)
game.add_available_button(vzd.Button.MOVE_RIGHT)
game.add_available_button(vzd.Button.MOVE_BACKWARD)
game.add_available_button(vzd.Button.MOVE_FORWARD)

game.set_window_visible(True)  # Ventana visible para ver al DQN en acción
game.set_mode(vzd.Mode.PLAYER)
game.set_screen_format(vzd.ScreenFormat.GRAY8)
game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)

game.add_available_game_variable(vzd.GameVariable.HEALTH)
game.add_available_game_variable(vzd.GameVariable.AMMO2)
game.add_available_game_variable(vzd.GameVariable.KILLCOUNT)
game.init()

print("\n==========================================================================")
print("  BASELINE DQN: MODO DEMO / EVALUACIÓN (PESOS CONGELADOS)")
print("==========================================================================\n")

FRAME_REPEAT = 2
PAUSA_PASO = 0.04  # Retardo visual para apreciar la partida
NUM_EPISODIOS_EVAL = 5

try:
    for ep in range(1, NUM_EPISODIOS_EVAL + 1):
        game.new_episode()
        step = 0

        while not game.is_episode_finished():
            state = game.get_state()
            screen = state.screen_buffer
            current_health = state.game_variables[0] if len(state.game_variables) > 0 else 100.0
            current_kills  = state.game_variables[2] if len(state.game_variables) > 2 else 0.0

            # Procesamiento visual idéntico (16x32)
            screen_t = torch.from_numpy(screen).float().to(device).unsqueeze(0).unsqueeze(0)
            retina = F.interpolate(screen_t, size=(16, 32), mode="area")
            retina_norm = (retina - retina.mean()) / (retina.std() + 1e-5)

            # Inferencia pura (sin exploración aleatoria)
            with torch.no_grad():
                q_values = policy_net(retina_norm)
                action_idx = q_values.max(1)[1].item()

            action = ACTIONS[action_idx]
            game.make_action(action, FRAME_REPEAT)
            step += 1

            time.sleep(PAUSA_PASO)

            if step % 50 == 0:
                print(f"Eval DQN Ep {ep} | Paso {step:03d} | HP: {current_health:4.1f} | Kills: {current_kills:2.0f}")

        print(f"--> Fin de Evaluación DQN Episodio {ep} | Kills Totales: {current_kills}\n")

except KeyboardInterrupt:
    print("\n--> Demostración DQN detenida manualmente.")
finally:
    game.close()
    print("Ventana de evaluación DQN cerrada correctamente.")
