import cv2
import numpy as np
import os
from collections import deque

SOG_SGG_ENABLED = True
SOG_THRESHOLD = 0.05
SGG_PROXIMITY_M = 5.0
SOG_ACTIVE_SPEED_KMH = 7.0
HISTORY_LENGTH = 10
INFLUENCE_STD_DEV = 7.0
EPV_ENABLED = True
EPV_INCREASE_THRESHOLD = 0.05
VALUE_RUN_TRAIL_LENGTH = 15

def load_real_coords_with_color(file_path):
    data = {}
    if not os.path.exists(file_path):
        print(f"Error: El archivo de coordenadas reales no se encontró en {file_path}")
        return data
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            parts = line.strip().split(',')
            if len(parts) == 5:
                try:
                    frame_id, player_id, x, y = int(parts[0]), int(parts[1]), float(parts[2]), float(parts[3])
                    color_parts = parts[4].split('-')
                    color = (int(color_parts[0]), int(color_parts[1]), int(color_parts[2]))
                    if frame_id not in data: data[frame_id] = []
                    data[frame_id].append({'player_id': player_id, 'x': x, 'y': y, 'color': color})
                except (ValueError, IndexError): continue
    return data

def load_det_data(file_path):
    detections = {}
    if not os.path.exists(file_path):
        print(f"Error: No se pudo encontrar el archivo en {file_path}"); return detections
    with open(file_path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            try:
                parts = line.strip().split(',')
                if len(parts) >= 6:
                    frame_id, player_id = int(parts[0]), int(parts[1])
                    if frame_id not in detections: detections[frame_id] = []
                    detections[frame_id].append({'player_id': player_id, 'bbox_x': float(parts[2]), 'bbox_y': float(parts[3]), 'bbox_w': float(parts[4]), 'bbox_h': float(parts[5]), 'numbers': []})
            except (ValueError, IndexError): continue
    return detections

def resize_and_pad(image, target_width):
    original_h, original_w = image.shape[:2]
    if original_w == target_width: return image
    return cv2.resize(image, (target_width, int(original_h * (target_width / original_w))))

def calculate_space_value_map(field_dims_m, output_dims_px, attack_direction='right'):
    field_length_m, _ = field_dims_m
    field_height_px, field_width_px = output_dims_px
    x_coords = np.linspace(0, 1, field_width_px)
    if attack_direction == 'left': x_coords = 1 - x_coords
    return np.tile(x_coords, (field_height_px, 1))

def calculate_pitch_control_per_team(players, field_dims_m, output_dims_px):
    field_length_m, field_width_m = field_dims_m
    field_height_px, field_width_px = output_dims_px
    x_grid, y_grid = np.mgrid[0:field_length_m:complex(field_height_px), 0:field_width_m:complex(field_width_px)]
    grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
    total_influence = np.zeros(grid_points.shape[0])
    for p in players:
        player_pos = np.array([p['y'], p['x']])
        sq_distances = np.sum((grid_points - player_pos)**2, axis=1)
        influence = np.exp(-sq_distances / (2 * (INFLUENCE_STD_DEV**2)))
        total_influence += influence
    return total_influence.reshape(field_height_px, field_width_px)

def calculate_sog_sgg_events(current_coords, prev_coords, space_value_map, fps, all_excluded_ids):
    if not prev_coords: return {}, {}
    sog_events, sgg_events = {}, []
    current_coords_filtered = [p for p in current_coords if p['player_id'] not in all_excluded_ids]
    prev_coords_filtered = [p for p in prev_coords if p['player_id'] not in all_excluded_ids]
    players_by_id_curr = {p['player_id']: p for p in current_coords_filtered}
    players_by_id_prev = {p['player_id']: p for p in prev_coords_filtered}
    team1_curr = [p for p in current_coords_filtered if 1 <= p['player_id'] <= 11]
    team2_curr = [p for p in current_coords_filtered if 12 <= p['player_id'] <= 22]
    pc_map_team1 = calculate_pitch_control_per_team(team1_curr, (105, 68), space_value_map.shape)
    pc_map_team2 = calculate_pitch_control_per_team(team2_curr, (105, 68), space_value_map.shape)
    for p_curr in current_coords_filtered:
        p_prev = players_by_id_prev.get(p_curr['player_id'])
        if not p_prev: continue
        h, w = space_value_map.shape
        x_px_curr, y_px_curr = min(w-1, int(p_curr['x']/105*w)), min(h-1, int(p_curr['y']/68*h))
        x_px_prev, y_px_prev = min(w-1, int(p_prev['x']/105*w)), min(h-1, int(p_prev['y']/68*h))
        pc_map_curr = pc_map_team1 if 1 <= p_curr['player_id'] <= 11 else pc_map_team2
        q_curr = pc_map_curr[y_px_curr, x_px_curr] * space_value_map[y_px_curr, x_px_curr]
        q_prev = pc_map_curr[y_px_prev, x_px_prev] * space_value_map[y_px_prev, x_px_prev]
        if (q_curr - q_prev) > SOG_THRESHOLD:
            dist_m = np.sqrt((p_curr['x'] - p_prev['x'])**2 + (p_curr['y'] - p_prev['y'])**2)
            speed_kmh = (dist_m / (HISTORY_LENGTH / fps)) * 3.6
            sog_events[p_curr['player_id']] = 'active' if speed_kmh > SOG_ACTIVE_SPEED_KMH else 'passive'
    return sog_events, sgg_events

def calculate_epv_simplified(coords, ball_carrier_id, all_excluded_ids, attacking_team_color=None, attack_direction='right'):
    if not coords or ball_carrier_id is None: return 0.0
    ball_carrier = next((p for p in coords if p['player_id'] == ball_carrier_id), None)
    if not ball_carrier: return 0.0
    field_length = 105.0
    ball_x = ball_carrier['x']
    positional_value = (ball_x / field_length) if attack_direction == 'right' else ((field_length - ball_x) / field_length)
    positional_value = np.clip(positional_value, 0, 1)
    if attacking_team_color is not None:
        attackers = [p for p in coords if p['color'] == attacking_team_color]
        defenders = [p for p in coords if p['color'] != attacking_team_color and p['player_id'] not in all_excluded_ids]
    else:
        is_team1_attacking = 1 <= ball_carrier_id <= 11
        attackers = [p for p in coords if (1 <= p['player_id'] <= 11 if is_team1_attacking else 12 <= p['player_id'] <= 22)]
        defenders = [p for p in coords if (12 <= p['player_id'] <= 22 if is_team1_attacking else 1 <= p['player_id'] <= 11)]
    attackers_ahead_of_ball = [p for p in attackers if p['x'] > ball_x]
    defenders_ahead_of_ball = [p for p in defenders if p['x'] > ball_x]
    superiority_factor = 1.0
    if len(defenders_ahead_of_ball) > 0:
        ratio = len(attackers_ahead_of_ball) / len(defenders_ahead_of_ball)
        superiority_factor = np.clip(ratio, 0.5, 1.5)
    elif len(attackers_ahead_of_ball) > 0:
        superiority_factor = 1.5
    final_epv = positional_value * 0.7 + (superiority_factor - 0.5) * 0.3
    return np.clip(final_epv, 0, 1)

def visualize_vertical_view(image_folder, detections_path, real_coords_path, fps=10, output_width=1280, exclude_ids=None, ball_ids=None, attacking_team_sample_ids=None):
    if exclude_ids is None: exclude_ids = []
    if ball_ids is None: ball_ids = []
    if attacking_team_sample_ids is None: attacking_team_sample_ids = []
    
    all_excluded_ids = set(exclude_ids + ball_ids)
    
    global SOG_SGG_ENABLED, EPV_ENABLED
    
    print("Cargando todos los datos...")
    image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if not image_files: print(f"No se encontraron imágenes en {image_folder}"); return
    
    PUNTOS_REALES = { "Esquina Izq-Inf": (0, 0), "Esquina Izq-Sup": (0, 68), "Esquina Der-Inf": (105, 0), "Esquina Der-Sup": (105, 68), "Centro Campo": (52.5, 34), "Centro-Banda Sup": (52.5, 68), "Centro-Banda Inf": (52.5, 0), "18m Izq-Inf Inter": (16.5, 13.85), "18m Izq-Sup Inter": (16.5, 54.15), "6m Izq-Inf Inter": (5.5, 24.84), "6m Izq-Sup Inter": (5.5, 43.16), "18m Der-Inf Inter": (88.5, 13.85), "18m Der-Sup Inter": (88.5, 54.15), "6m Der-Inf Inter": (99.5, 24.84), "6m Der-Sup Inter": (99.5, 43.16) }
    detections_data, real_coords_data = load_det_data(detections_path), load_real_coords_with_color(real_coords_path)
    
    attacking_team_color = None
    if attacking_team_sample_ids and real_coords_data:
        first_frame_id = next(iter(real_coords_data))
        for player_coord in real_coords_data[first_frame_id]:
            if player_coord['player_id'] in attacking_team_sample_ids:
                attacking_team_color = player_coord['color']
                print(f"Color del equipo atacante detectado: {attacking_team_color}")
                break

    field_template = draw_soccer_field(PUNTOS_REALES, output_width=output_width)
    field_dims_m = (105.0, 68.0)
    field_height_px, field_width_px = field_template.shape[:2]
    space_value_map = calculate_space_value_map(field_dims_m, (field_height_px, field_width_px), 'right')
    
    paused, frame_delay = False, int(1000 / fps)
    coords_history = deque(maxlen=HISTORY_LENGTH)
    
    epv_history = deque(maxlen=HISTORY_LENGTH)
    value_adding_runs = {}
    
    print("\n=== CONTROLES DE VISUALIZACIÓN ===")
    print("'q': Salir, ESPACIO: Pausar, +/-: Velocidad, 's': SOG/SGG, 'e': EPV")
    
    for image_file in image_files:
        frame_id = int(os.path.splitext(image_file)[0])
        original_frame = cv2.imread(os.path.join(image_folder, image_file))
        if original_frame is None: continue
        
        padded_frame = resize_and_pad(original_frame, target_width=output_width)
        field_2d_view = field_template.copy()
        
        current_coords = real_coords_data.get(frame_id, [])
        coords_history.append(current_coords)
        players_by_id_curr = {p['player_id']: p for p in current_coords}
        
        sog_events = {}
        if SOG_SGG_ENABLED and len(coords_history) == HISTORY_LENGTH:
            prev_coords = coords_history[0]
            sog_events, _ = calculate_sog_sgg_events(current_coords, prev_coords, space_value_map, fps, all_excluded_ids)
            
        ball_carrier_id = 1
        current_epv = calculate_epv_simplified(current_coords, ball_carrier_id, all_excluded_ids, attacking_team_color, 'right') if EPV_ENABLED else 0.0
        epv_history.append(current_epv)
        
        if len(epv_history) == HISTORY_LENGTH:
            epv_change = current_epv - epv_history[0]
            if epv_change > EPV_INCREASE_THRESHOLD:
                best_player_id, max_dx = -1, -1
                prev_coords = coords_history[0]
                players_by_id_prev = {p['player_id']: p for p in prev_coords}
                for p_curr in current_coords:
                    p_prev = players_by_id_prev.get(p_curr['player_id'])
                    if p_prev and p_curr['color'] == attacking_team_color:
                        dx = p_curr['x'] - p_prev['x']
                        if dx > max_dx:
                            max_dx, best_player_id = dx, p_curr['player_id']
                if best_player_id != -1:
                    if best_player_id not in value_adding_runs:
                        value_adding_runs[best_player_id] = {'trail': deque(maxlen=VALUE_RUN_TRAIL_LENGTH), 'timer': 0}
                    value_adding_runs[best_player_id]['timer'] = VALUE_RUN_TRAIL_LENGTH

        for player_id, data in list(value_adding_runs.items()):
            player_pos = players_by_id_curr.get(player_id)
            if player_pos:
                x_px = int(player_pos['x']/field_dims_m[0]*field_width_px)
                y_px = int(player_pos['y']/field_dims_m[1]*field_height_px)
                data['trail'].append((x_px, y_px))
            if len(data['trail']) > 1:
                cv2.polylines(field_2d_view, [np.array(data['trail'], dtype=np.int32)], False, (255, 0, 255), 2)
            data['timer'] -= 1
            if data['timer'] <= 0: del value_adding_runs[player_id]

        for coord_info in current_coords:
            p_id, x_m, y_m, p_color = coord_info['player_id'], coord_info['x'], coord_info['y'], coord_info['color']
            x_px, y_px = int(x_m / field_dims_m[0] * field_width_px), int(y_m / field_dims_m[1] * field_height_px)
            
            if p_id in ball_ids:
                cv2.circle(field_2d_view, (x_px, y_px), 4, (255, 255, 255), -1)
                cv2.circle(field_2d_view, (x_px, y_px), 4, (0, 0, 0), 1)
            elif p_id in exclude_ids:
                cv2.circle(field_2d_view, (x_px, y_px), 5, p_color, 1)
                cv2.putText(field_2d_view, str(p_id), (x_px + 8, y_px + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            else:
                if p_id in sog_events:
                    aura_color = (0, 255, 255) if sog_events[p_id] == 'active' else (255, 255, 0)
                    cv2.circle(field_2d_view, (x_px, y_px), 10, aura_color, 1)
                
                cv2.circle(field_2d_view, (x_px, y_px), 6, p_color, -1)
                cv2.circle(field_2d_view, (x_px, y_px), 6, (0, 0, 0), 1)
                cv2.putText(field_2d_view, str(p_id), (x_px + 10, y_px + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if EPV_ENABLED:
            bar_width = int(padded_frame.shape[1] * 0.4)
            bar_x, bar_y = (padded_frame.shape[1] - bar_width) // 2, 60
            bar_color_b, bar_color_r = int(255 * (1 - current_epv)), int(255 * current_epv)
            cv2.rectangle(padded_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), (50, 50, 50), -1)
            cv2.rectangle(padded_frame, (bar_x, bar_y), (bar_x + int(bar_width * current_epv), bar_y + 20), (bar_color_b, 80, bar_color_r), -1)
            cv2.rectangle(padded_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), (255, 255, 255), 1)
            cv2.putText(padded_frame, "VALOR DE POSESION (EPV)", (bar_x, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        status_text_sog = f"SOG/SGG: {'ON' if SOG_SGG_ENABLED else 'OFF'}"
        status_text_epv = f"EPV: {'ON' if EPV_ENABLED else 'OFF'}"
        cv2.putText(padded_frame, f"Frame: {frame_id} | {status_text_sog} | {status_text_epv}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        combined_view = np.vstack((padded_frame, field_2d_view))
        cv2.imshow("Vista Combinada: Partido (Arriba) vs Analisis Tactico 2D (Abajo)", combined_view)
        
        while True:
            key = cv2.waitKey(1 if paused else frame_delay) & 0xFF
            if key == ord('q'): cv2.destroyAllWindows(); return
            elif key == ord(' '): paused = not paused; break
            elif key == ord('s'): SOG_SGG_ENABLED = not SOG_SGG_ENABLED; break
            elif key == ord('e'): EPV_ENABLED = not EPV_ENABLED; break
            elif key == ord('+'): fps = min(fps + 5, 60); frame_delay = int(1000/fps); break
            elif key == ord('-'): fps = max(fps - 5, 1); frame_delay = int(1000/fps); break
            elif not paused: break
            
    cv2.destroyAllWindows()

def draw_soccer_field(puntos_reales_dict, output_width=1280):
    field_length_m, field_width_m = 105.0, 68.0
    aspect_ratio = field_width_m / field_length_m
    output_height = int(output_width * aspect_ratio)
    field = np.full((output_height, output_width, 3), (20, 110, 25), dtype=np.uint8)
    line_color, line_thickness = (255, 255, 255), 2
    def scale(coords):
        x_m, y_m = coords; return int(x_m/field_length_m*output_width), int(y_m/field_width_m*output_height)
    cv2.rectangle(field, scale(puntos_reales_dict["Esquina Izq-Inf"]), scale(puntos_reales_dict["Esquina Der-Sup"]), line_color, line_thickness)
    cv2.line(field, scale(puntos_reales_dict["Centro-Banda Inf"]), scale(puntos_reales_dict["Centro-Banda Sup"]), line_color, line_thickness)
    radius_px = int(9.15/field_length_m*output_width)
    cv2.circle(field, scale(puntos_reales_dict["Centro Campo"]), radius_px, line_color, line_thickness)
    cv2.rectangle(field, scale(puntos_reales_dict["18m Izq-Inf Inter"]), scale((0, puntos_reales_dict["18m Izq-Sup Inter"][1])), line_color, line_thickness)
    cv2.rectangle(field, scale(puntos_reales_dict["18m Der-Inf Inter"]), scale((105, puntos_reales_dict["18m Der-Sup Inter"][1])), line_color, line_thickness)
    cv2.rectangle(field, scale(puntos_reales_dict["6m Izq-Inf Inter"]), scale((0, puntos_reales_dict["6m Izq-Sup Inter"][1])), line_color, line_thickness)
    cv2.rectangle(field, scale(puntos_reales_dict["6m Der-Inf Inter"]), scale((105, puntos_reales_dict["6m Der-Sup Inter"][1])), line_color, line_thickness)
    cv2.ellipse(field, scale((11, 34)), (radius_px, radius_px), 0, -53, 53, line_color, line_thickness)
    cv2.ellipse(field, scale((94, 34)), (radius_px, radius_px), 0, 127, 233, line_color, line_thickness)
    return field

if __name__ == '__main__':
    base_dir = "Ruta/A/La/Carpeta/De/La/Secuencia"
    ids_a_excluir = [17, 20, 24]
    ids_balon = [22]
    ids_equipo_atacante_muestra = [14,15]

    visualize_vertical_view(
        image_folder=os.path.join(base_dir, "img1"),
        detections_path=os.path.join(base_dir, "det/sorted_det.txt"),
        real_coords_path=os.path.join(base_dir, "CoordPuntosRealesColor.txt"),
        fps=10,
        output_width=800,
        exclude_ids=ids_a_excluir,
        ball_ids=ids_balon,
        attacking_team_sample_ids=ids_equipo_atacante_muestra
    )
