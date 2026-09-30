import os
import time
import numpy as np
import torch
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
print(f"--> Dispositivo activo (Modo Demo PURAMENTE LÓGICO / SIN CONECTOME): {device} ({dev_name})")

# ==========================================
# 2. CONFIGURACIÓN VIZDOOM (MODO VISIBLE)
# ==========================================
game = vzd.DoomGame()
scenarios_dir = vzd.scenarios_path
game.load_config(os.path.join(scenarios_dir, "defend_the_center.cfg"))

game.add_available_button(vzd.Button.ATTACK)
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
print("  AGENTE PURAMENTE HEURÍSTICO (SIN RED BIOLÓGICA)")
print("==========================================================================\n")

NUM_EPISODIOS_EVAL = 5
NUM_SECTORS = 8

try:
    for ep in range(1, NUM_EPISODIOS_EVAL + 1):
        game.new_episode()
        step = 0

        shotgun_cooldown = 0
        refractory_damage = 0
        pain_turn_ticks = 0
        pain_turn_dir = 0
        last_kills = 0

        init_state = game.get_state()
        last_health = init_state.game_variables[0] if (init_state and len(init_state.game_variables) > 0) else 100.0

        while not game.is_episode_finished():
            state = game.get_state()
            screen = state.screen_buffer
            current_health = state.game_variables[0] if len(state.game_variables) > 0 else 100.0
            current_kills  = state.game_variables[2] if len(state.game_variables) > 2 else last_kills

            screen_t = torch.from_numpy(screen).float().to(device).unsqueeze(0).unsqueeze(0)
            retina = F.interpolate(screen_t, size=(16, 32), mode="area").squeeze()

            retina_norm = (retina - retina.mean()) / (retina.std() + 1e-5)
            contrast = torch.clamp(torch.abs(retina_norm) / 2.5, 0.0, 1.0)

            health_delta = current_health - last_health
            if health_delta < 0 and refractory_damage == 0:
                pain_turn_ticks = 4
                refractory_damage = 6
                pain_turn_dir = 1 if np.random.rand() > 0.5 else 2

            if refractory_damage > 0:
                refractory_damage -= 1

            combat_band = contrast[2:9, :]
            horizontal_profile = combat_band.max(dim=0).values.cpu().numpy()
            target_col = int(np.argmax(horizontal_profile))
            max_salience = float(horizontal_profile[target_col])
            has_real_target = max_salience > 0.35

            sector_vals = []
            for s in range(NUM_SECTORS):
                col_start = s * 4
                col_end = (s + 1) * 4
                s_val = float(combat_band[:, col_start:col_end].max().item())
                sector_vals.append(s_val)

            val_l = float(np.mean(sector_vals[:3]))
            val_r = float(np.mean(sector_vals[5:]))

            action = [0, 0, 0, 0, 0]
            act_tag = "PATRULLA"

            if shotgun_cooldown > 0:
                shotgun_cooldown -= 1

            if pain_turn_ticks > 0:
                action[pain_turn_dir] = 1
                action[3] = 1
                pain_turn_ticks -= 1
                act_tag = "REPRESALIA_IZQ" if pain_turn_dir == 1 else "REPRESALIA_DER"
            elif has_real_target and shotgun_cooldown == 0:
                if 11 <= target_col <= 21:
                    action[0] = 1  # ATTACK
                    shotgun_cooldown = 3
                    act_tag = "FUEGO_QUEMARROPA_INSTANTANEO" if max_salience > 0.6 else "FUEGO_CONFIRMADO"
                elif target_col < 11:
                    action[1] = 1  # MOVE_LEFT
                    act_tag = "CAZA_IZQ"
                else:
                    action[2] = 1  # MOVE_RIGHT
                    act_tag = "CAZA_DER"
            else:
                diff_lr = val_l - val_r
                if diff_lr > 0.08:
                    action[1] = 1
                    act_tag = "BUSQUEDA_IZQ"
                elif diff_lr < -0.08:
                    action[2] = 1
                    act_tag = "BUSQUEDA_DER"
                else:
                    action[4] = 1  # MOVE_FORWARD
                    act_tag = "BARRIDO_INERCIAL"

            game.make_action(action, 2)
            step += 1
            last_health = current_health
            last_kills = current_kills
            time.sleep(0.04)

            if step % 20 == 0:
                print(f"Eval Ep {ep} | Paso {step:03d} | [SIN RED BIOLÓGICA] | HP:{current_health:.1f} | Kills: {current_kills:.0f} -> {act_tag}")

        print(f"--> Fin de Evaluación Episodio {ep} | Kills Totales: {last_kills:.1f}\n")

except KeyboardInterrupt:
    print("\n--> Demostración detenida.")
finally:
    game.close()
    print("Demo cerrada.")
