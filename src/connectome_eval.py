import os
import time
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import vizdoom as vzd

# ==========================================
# 1. DISPOSITIVO Y GPU
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--> Dispositivo activo (Modo Evaluación): {device} ({torch.cuda.get_device_name(0)})")

# ==========================================
# 2. CARGA DEL CONECTOMA BIOLÓGICO (CSR)
# ==========================================
npz_file = "flywire_female_brain.npz"
adj_scipy = sp.load_npz(npz_file).tocsr()
n_neurons = adj_scipy.shape[0]

indptr = torch.tensor(adj_scipy.indptr, dtype=torch.int64, device=device)
indices = torch.tensor(adj_scipy.indices, dtype=torch.int64, device=device)
raw_weights = torch.tensor(adj_scipy.data, dtype=torch.float32, device=device)

weights = torch.clamp(raw_weights * 0.005, 0.0, 0.20)
W_bio = torch.sparse_csr_tensor(indptr, indices, weights, size=(n_neurons, n_neurons), device=device)
del adj_scipy, raw_weights

# ==========================================
# 3. PUERTOS ANATÓMICOS RETINOTÓPICOS (8 SECTORES)
# ==========================================
vis_idx_np = np.load("indices_visuales.npy")
mot_idx_np = np.load("indices_motores.npy")

N_IN = len(vis_idx_np)
print(f"--> Neuronas sensoriales visuales activas: {N_IN}")

NUM_SECTORS = 8
split_v = N_IN // NUM_SECTORS
sens_sectors = []
for s in range(NUM_SECTORS):
    start = s * split_v
    end = (s + 1) * split_v if s < NUM_SECTORS - 1 else N_IN
    sens_sectors.append(torch.tensor(vis_idx_np[start:end], device=device, dtype=torch.long))

n_mot = len(mot_idx_np)
split_m = n_mot // 7
mot_turn_l   = torch.tensor(mot_idx_np[:split_m], device=device, dtype=torch.long)
mot_turn_r   = torch.tensor(mot_idx_np[split_m:2*split_m], device=device, dtype=torch.long)
mot_fire     = torch.tensor(mot_idx_np[2*split_m:3*split_m], device=device, dtype=torch.long)
mot_strafe_l = torch.tensor(mot_idx_np[3*split_m:4*split_m], device=device, dtype=torch.long)
mot_strafe_r = torch.tensor(mot_idx_np[4*split_m:5*split_m], device=device, dtype=torch.long)
mot_back     = torch.tensor(mot_idx_np[5*split_m:6*split_m], device=device, dtype=torch.long)
mot_forward  = torch.tensor(mot_idx_np[6*split_m:], device=device, dtype=torch.long)

# CARGA DE PESOS ENTRENADOS (CONGELADOS)
##WEIGHTS_SAVE_FILE = "pesos_plasticos_v22.pt"
WEIGHTS_SAVE_FILE = "pesos_plasticos_v23.pt"
if os.path.exists(WEIGHTS_SAVE_FILE):
    w_plastic = torch.load(WEIGHTS_SAVE_FILE, map_location=device, weights_only=True)
    print(f"--> Pesos entrenados cargados correctamente desde: {WEIGHTS_SAVE_FILE}")
    print(f"--> Ganancia de Fuego actual congelada: {w_plastic[2, 2]:.2f}")
else:
    raise FileNotFoundError(f"No se encontró el archivo de pesos '{WEIGHTS_SAVE_FILE}'. ¡Debes entrenarlo primero!")

# ==========================================
# 4. PARÁMETROS BIOFÍSICOS Y DINÁMICOS
# ==========================================
TAU_M = 0.82
V_REST = 0.0
V_RESET = -0.10
V_THRESH = 1.0
INHIBITION_K = 0.0005  # Freno anti-saturación

FRAME_REPEAT = 2
PAUSA_PASO   = 0.04    # Retardo visual para poder apreciar la partida en pantalla

HABITUATION_DECAY = 0.60
scale_factor = 6000.0 / float(N_IN)

# ==========================================
# 5. CONFIGURACIÓN VIZDOOM (MODO VISIBLE)
# ==========================================
game = vzd.DoomGame()
scenarios_dir = vzd.scenarios_path
game.load_config(os.path.join(scenarios_dir, "defend_the_center.cfg"))

game.add_available_button(vzd.Button.MOVE_LEFT)
game.add_available_button(vzd.Button.MOVE_RIGHT)
game.add_available_button(vzd.Button.MOVE_BACKWARD)
game.add_available_button(vzd.Button.MOVE_FORWARD)

game.set_window_visible(True)  # Ventana visible para demostración en vivo
game.set_mode(vzd.Mode.PLAYER)
game.set_screen_format(vzd.ScreenFormat.GRAY8)
game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)

game.add_available_game_variable(vzd.GameVariable.HEALTH)
game.add_available_game_variable(vzd.GameVariable.AMMO2)
game.add_available_game_variable(vzd.GameVariable.KILLCOUNT)
game.init()

print("\n==========================================================================")
print("  CONECTOMA V22.9: MODO DEMO / EVALUACIÓN (PESOS CONGELADOS)")
print("==========================================================================\n")

NUM_EPISODIOS_EVAL = 5

try:
    for ep in range(1, NUM_EPISODIOS_EVAL + 1):
        game.new_episode()
        v_m = torch.full((n_neurons, 1), V_REST, device=device)
        spikes = torch.zeros((n_neurons, 1), device=device)
        step = 0
        base_tl, base_tr, base_f = 1.0, 1.0, 1.0
        base_sl, base_sr, base_bk, base_fw = 1.0, 1.0, 1.0, 1.0
        fovea_fatigue = 0.0
        shotgun_cooldown = 0

        turn_persistence = 0
        active_turn_dir = 0
        last_patrol_dir = 0

        prev_retina = None
        wall_avoid_ticks = 0
        wall_escape_turn = 1
        wall_escape_strafe = 4
        pain_turn_ticks = 0
        pain_turn_dir = 0
        refractory_damage = 0
        action_prev_fire = False

        init_state = game.get_state()
        last_health = init_state.game_variables[0] if (init_state and len(init_state.game_variables) > 0) else 100.0
        last_ammo   = init_state.game_variables[1] if (init_state and len(init_state.game_variables) > 1) else 50.0
        last_kills  = init_state.game_variables[2] if (init_state and len(init_state.game_variables) > 2) else 0.0

        while not game.is_episode_finished():
            state = game.get_state()
            screen = state.screen_buffer
            current_health = state.game_variables[0] if len(state.game_variables) > 0 else 100.0
            current_ammo   = state.game_variables[1] if len(state.game_variables) > 1 else last_ammo
            current_kills  = state.game_variables[2] if len(state.game_variables) > 2 else last_kills

            screen_t = torch.from_numpy(screen).float().to(device).unsqueeze(0).unsqueeze(0)
            retina = F.interpolate(screen_t, size=(16, 32), mode="area").squeeze()

            retina_norm = (retina - retina.mean()) / (retina.std() + 1e-5)
            contrast = torch.clamp(torch.abs(retina_norm) / 2.5, 0.0, 1.0)

            health_delta = current_health - last_health

            if health_delta < 0:
                prev_retina = contrast.clone()

            if prev_retina is None:
                prev_retina = contrast.clone()

            delta_temp = torch.abs(contrast - prev_retina)

            top_band_mean = float(contrast[:5, :].mean().item())
            mid_band_mean = float(contrast[5:11, :].mean().item())

            top_c = float(contrast[:5, 11:21].mean().item())
            mid_c = float(contrast[5:11, 11:21].mean().item())
            is_front_wall = (top_c > 0.50) and (mid_c > 0.48) and (abs(top_c - mid_c) < 0.10)
            is_stuck_wall = (top_band_mean > 0.52) and (mid_band_mean > 0.52)

            wall_left_val  = float(contrast[:, :10].mean().item())
            wall_right_val = float(contrast[:, 22:].mean().item())

            if is_stuck_wall and wall_avoid_ticks == 0:
                wall_avoid_ticks = 5
                if wall_left_val >= wall_right_val:
                    wall_escape_turn   = 1
                    wall_escape_strafe = 4
                else:
                    wall_escape_turn   = 0
                    wall_escape_strafe = 3

            motion_component = contrast * (1.0 + 2.0 * delta_temp)
            hybrid_map = 0.50 * contrast + 0.50 * motion_component
            prev_retina = contrast.clone()

            combat_band = hybrid_map[2:9, :]
            horizontal_profile = combat_band.max(dim=0).values.cpu().numpy()

            target_col = int(np.argmax(horizontal_profile))
            max_salience = float(horizontal_profile[target_col])
            baseline_salience = float(np.median(horizontal_profile))
            has_real_target = (max_salience - baseline_salience > 0.16) and (max_salience > 0.35)

            fovea_zone = contrast[2:9, 11:21]
            target_mass = float(fovea_zone.mean().item())
            pinky_zone  = float(contrast[6:10, 11:21].mean().item())

            sector_vals = []
            for s in range(NUM_SECTORS):
                col_start = s * 4
                col_end = (s + 1) * 4
                s_val = float(combat_band[:, col_start:col_end].max().item() * 0.7 +
                              combat_band[:, col_start:col_end].mean().item() * 0.3)
                sector_vals.append(s_val)

            val_l = float(np.mean(sector_vals[:3]))
            val_c = float(np.mean(sector_vals[3:5]))
            val_r = float(np.mean(sector_vals[5:]))
            delta_foveal = float(delta_temp[2:9, 11:21].max().item())

            if health_delta < 0 and refractory_damage == 0:
                pain_turn_ticks = 4
                refractory_damage = 6
                turn_persistence = 0
                if val_l >= val_r:
                    pain_turn_dir = 0
                    last_patrol_dir = 0
                else:
                    pain_turn_dir = 1
                    last_patrol_dir = 1

            if refractory_damage > 0:
                refractory_damage -= 1

            if delta_foveal > 0.08:
                fovea_fatigue = 0.0
            elif target_mass > 0.20 and action_prev_fire:
                fovea_fatigue += 0.12
            else:
                fovea_fatigue *= HABITUATION_DECAY

            fovea_gain = max(0.35, 1.0 - (fovea_fatigue * 0.25))
            val_c_efectivo = val_c * fovea_gain

            retina_act = torch.tensor([val_l, val_r, val_c_efectivo], dtype=torch.float32, device=device)

            # Propagación pura con pesos congelados (sin R-STDP)
            gains = torch.matmul(retina_act, w_plastic)
            gain_tl, gain_tr, gain_f = gains[0].item(), gains[1].item(), gains[2].item()

            i_ext = torch.zeros((n_neurons, 1), device=device)
            for s in range(NUM_SECTORS):
                if s < 3:
                    g = gain_tl * 6.5
                elif s > 4:
                    g = gain_tr * 6.5
                else:
                    g = gain_f * 12.0 * fovea_gain

                i_ext[sens_sectors[s]] = sector_vals[s] * g * scale_factor

            i_syn = torch.matmul(W_bio, spikes)
            i_inhib = spikes.sum() * INHIBITION_K

            v_m = (v_m * TAU_M) + i_syn + i_ext - i_inhib
            spikes = (v_m >= V_THRESH).float()
            v_m = torch.where(spikes > 0, torch.full_like(v_m, V_RESET), v_m)

            total_sp = int(spikes.sum().item())

            motor_counts = torch.stack([
                spikes[mot_turn_l].sum(),
                spikes[mot_turn_r].sum(),
                spikes[mot_fire].sum(),
                spikes[mot_strafe_l].sum(),
                spikes[mot_strafe_r].sum(),
                spikes[mot_back].sum(),
                spikes[mot_forward].sum()
            ]).cpu().numpy()

            raw_tl, raw_tr, raw_f, raw_sl, raw_sr, raw_bk, raw_fw = motor_counts

            base_tl = 0.94 * base_tl + 0.06 * raw_tl
            base_tr = 0.94 * base_tr + 0.06 * raw_tr
            base_f  = 0.94 * base_f  + 0.06 * raw_f
            base_sl = 0.94 * base_sl + 0.06 * raw_sl
            base_sr = 0.94 * base_sr + 0.06 * raw_sr
            base_bk = 0.94 * base_bk + 0.06 * raw_bk
            base_fw = 0.94 * base_fw + 0.06 * raw_fw

            dtl = raw_tl - base_tl
            dtr = raw_tr - base_tr
            df  = raw_f  - base_f
            dsl = raw_sl - base_sl
            dsr = raw_sr - base_sr

            action = [0, 0, 0, 0, 0, 0, 0]
            act_tags = []

            if shotgun_cooldown > 0:
                shotgun_cooldown -= 1

            if wall_avoid_ticks > 0:
                action[5] = 1
                action[wall_escape_strafe] = 1
                action[wall_escape_turn]   = 1
                wall_avoid_ticks -= 1
                act_tags.append("DESPEJE_VECTORIAL")
                action_prev_fire = False
                turn_persistence = 0

            elif pain_turn_ticks > 0:
                action[pain_turn_dir] = 1
                action[5] = 1
                pain_turn_ticks -= 1
                act_tags.append("REPRESALIA_IZQ" if pain_turn_dir == 0 else "REPRESALIA_DER")
                action_prev_fire = False
                turn_persistence = 0

            elif current_health > 0:
                target_centered = (11 <= target_col <= 21) and has_real_target
                pinky_at_feet = (pinky_zone > 0.25) and (top_c < 0.45)

                near_center_zone = (6 <= target_col <= 26) and has_real_target
                is_point_blank = (target_mass > 0.28) and near_center_zone

                valid_enemy = (target_centered or is_point_blank or pinky_at_feet) and (not is_front_wall) and (not is_stuck_wall)

                motor_ready = (df > 0.10) or (raw_f > 0.4) or is_point_blank
                allow_fire = valid_enemy and motor_ready and (shotgun_cooldown == 0)

                if allow_fire:
                    action[2] = 1
                    act_tags.append("FUEGO_CONFIRMADO" if not is_point_blank else "FUEGO_QUEMARROPA_INSTANTANEO")
                    action_prev_fire = True
                    shotgun_cooldown = 3
                    turn_persistence = 0
                    action[0] = 0
                    action[1] = 0
                    action[6] = 0
                else:
                    action_prev_fire = False

                    if has_real_target:
                        if target_col < 11:
                            action[0] = 1
                            act_tags.append("CAZA_IZQ")
                            active_turn_dir = 0
                            turn_persistence = 3
                            last_patrol_dir = 0
                        elif target_col > 21:
                            action[1] = 1
                            act_tags.append("CAZA_DER")
                            active_turn_dir = 1
                            turn_persistence = 3
                            last_patrol_dir = 1
                        else:
                            turn_persistence = 0
                            if target_mass < 0.35 and not is_stuck_wall:
                                action[6] = 1
                                act_tags.append("AVANCE_RECTO")
                    else:
                        if turn_persistence > 0:
                            action[active_turn_dir] = 1
                            turn_persistence -= 1
                            act_tags.append("BARRIDO_INERCIAL")
                        else:
                            diff_lr = val_l - val_r
                            diff_dt = dtl - dtr

                            if (diff_lr > 0.08) or (diff_dt > 0.25):
                                action[0] = 1
                                active_turn_dir = 0
                                turn_persistence = 4
                                last_patrol_dir = 0
                                act_tags.append("BUSQUEDA_IZQ")
                            elif (diff_lr < -0.08) or (diff_dt < -0.25):
                                action[1] = 1
                                active_turn_dir = 1
                                turn_persistence = 4
                                last_patrol_dir = 1
                                act_tags.append("BUSQUEDA_DER")
                            else:
                                action[last_patrol_dir] = 1
                                act_tags.append("PATRULLA")

                    if action[6] == 0 and not is_stuck_wall:
                        if dsl > dsr + 0.5:
                            action[3] = 1
                            act_tags.append("PASO_IZQ")
                        elif dsr > dsl + 0.5:
                            action[4] = 1
                            act_tags.append("PASO_DER")

            game.make_action(action, FRAME_REPEAT)
            step += 1

            last_health = current_health
            last_kills = current_kills

            act_name = "+".join(act_tags) if act_tags else "ESPERA"
            time.sleep(PAUSA_PASO)  # Pausa visual para ver la simulación en pantalla

            if step % 20 == 0:
                print(f"Eval Ep {ep} | Paso {step:03d} | Espigas: {total_sp:5d} ({total_sp/n_neurons*100:4.1f}%) | "
                      f"HP:{current_health:4.1f} | Kills:{current_kills:2.0f} -> {act_name}")

        print(f"--> Fin de Evaluación Episodio {ep} | Kills Totales: {last_kills}\n")

except KeyboardInterrupt:
    print("\n--> Demostración detenida manualmente.")
finally:
    game.close()
    print("Ventana de evaluación cerrada correctamente.")
