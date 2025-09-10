import cv2
import numpy as np
import os
from collections import deque

OFF_BALL_ADV_ENABLED = True
ADVANTAGE_PRESSURE_RADIUS_M = 5.0 
ADVANTAGE_POSITIONAL_THRESHOLD = 0.6 

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

def calculate_off_ball_advantage(coords, ball_carrier_id, all_excluded_ids, attacking_team_color=None):
    if not coords or ball_carrier_id is None: return []
    
    ball_carrier = next((p for p in coords if p['player_id'] == ball_carrier_id), None)
    if not ball_carrier: return []
    valid_coords = [p for p in coords if 0 <= p['x'] <= 105 and 0 <= p['y'] <= 68]
    
    if attacking_team_color is not None:
        attacking_players = [p for p in valid_coords if p['color'] == attacking_team_color]
        defending_players = [p for p in valid_coords if p['color'] != attacking_team_color and p['player_id'] not in all_excluded_ids]
    else:
        attacking_team_id = 1 if 1 <= ball_carrier_id <= 11 else 2
        attacking_players = [p for p in valid_coords if (1 <= p['player_id'] <= 11 if attacking_team_id == 1 else 12 <= p['player_id'] <= 22)]
        defending_players = [p for p in valid_coords if (12 <= p['player_id'] <= 22 if attacking_team_id == 1 else 1 <= p['player_id'] <= 11)]
    
    advantage_players = []
    
    for attacker in attacking_players:
        if attacker['player_id'] == ball_carrier_id: continue

        is_in_dangerous_zone = (attacker['x'] / 105.0) < (1 - ADVANTAGE_POSITIONAL_THRESHOLD)
        
        if is_in_dangerous_zone:
            pressure_count = 0
            for defender in defending_players:
                dist = np.sqrt((attacker['x'] - defender['x'])**2 + (attacker['y'] - defender['y'])**2)
                if dist < ADVANTAGE_PRESSURE_RADIUS_M:
                    pressure_count += 1
            
            if pressure_count <= 1:
                advantage_players.append(attacker['player_id'])
                
    return advantage_players

def visualize_vertical_view(image_folder, detections_path, real_coords_path, fps=10, output_width=1280, exclude_ids=None, ball_ids=None, attacking_team_sample_ids=None):
    if exclude_ids is None: exclude_ids = []
    if ball_ids is None: ball_ids = []
    if attacking_team_sample_ids is None: attacking_team_sample_ids = []
    
    all_excluded_ids = set(exclude_ids + ball_ids)
    
    global OFF_BALL_ADV_ENABLED
    
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
    
    paused, frame_delay = False, int(1000 / fps)
    
    print("\n=== CONTROLES DE VISUALIZACIÓN ===")
    print("'q': Salir, ESPACIO: Pausar/Reanudar, +/-: Velocidad")
    print("'.': Frame siguiente (en pausa), ',': Frame anterior (en pausa)")
    print("'a': Off-Ball Advantage")
    
    frame_idx = 0
    while frame_idx < len(image_files):
        image_file = image_files[frame_idx]
        frame_id = int(os.path.splitext(image_file)[0])
        original_frame = cv2.imread(os.path.join(image_folder, image_file))
        if original_frame is None:
            frame_idx += 1
            continue
        
        padded_frame = resize_and_pad(original_frame, target_width=output_width)
        field_2d_view = field_template.copy()
        
        current_coords = real_coords_data.get(frame_id, [])
        players_by_id_curr = {p['player_id']: p for p in current_coords}
        
        ball_carrier_id = None
        if current_coords:
            for coord in current_coords:
                if (coord['color'] == attacking_team_color and 
                    coord['player_id'] not in all_excluded_ids and 
                    0 <= coord['x'] <= 105 and 0 <= coord['y'] <= 68):
                    ball_carrier_id = coord['player_id']
                    break
            
            if ball_carrier_id is None:
                for coord in current_coords:
                    if (coord['player_id'] not in all_excluded_ids and 
                        0 <= coord['x'] <= 105 and 0 <= coord['y'] <= 68):
                        ball_carrier_id = coord['player_id']
                        break

        advantage_players = []
        if OFF_BALL_ADV_ENABLED:
            advantage_players = calculate_off_ball_advantage(current_coords, ball_carrier_id, all_excluded_ids, attacking_team_color)

        for coord_info in current_coords:
            p_id, x_m, y_m, p_color = coord_info['player_id'], coord_info['x'], coord_info['y'], coord_info['color']
            
            x_px = max(0, min(field_width_px - 1, int(x_m / field_dims_m[0] * field_width_px)))
            y_px = max(0, min(field_height_px - 1, int(y_m / field_dims_m[1] * field_height_px)))
            
            if not (0 <= x_m <= 105 and 0 <= y_m <= 68):
                continue
            
            if p_id in ball_ids:
                cv2.circle(field_2d_view, (x_px, y_px), 4, (255, 255, 255), -1)
                cv2.circle(field_2d_view, (x_px, y_px), 4, (0, 0, 0), 1)
            elif p_id in exclude_ids:
                cv2.circle(field_2d_view, (x_px, y_px), 5, p_color, 1)
                cv2.putText(field_2d_view, str(p_id), (x_px + 8, y_px + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            else:
                if OFF_BALL_ADV_ENABLED and p_id in advantage_players:
                    cv2.circle(field_2d_view, (x_px, y_px), 12, (255, 255, 255), 2)
                
                cv2.circle(field_2d_view, (x_px, y_px), 6, p_color, -1)
                cv2.circle(field_2d_view, (x_px, y_px), 6, (0, 0, 0), 1)
                cv2.putText(field_2d_view, str(p_id), (x_px + 10, y_px + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        status_text = f"Frame: {frame_id} ({frame_idx+1}/{len(image_files)}) | FPS: {fps} | OffBall: {'ON' if OFF_BALL_ADV_ENABLED else 'OFF'}{' [PAUSADO]' if paused else ''}"
        cv2.putText(padded_frame, status_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)
        cv2.putText(padded_frame, status_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        combined_view = np.vstack((padded_frame, field_2d_view))
        cv2.imshow("Vista Combinada: Partido (Arriba) vs Analisis Tactico 2D (Abajo)", combined_view)
        
        key = cv2.waitKey(0 if paused else frame_delay) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('a'):
            OFF_BALL_ADV_ENABLED = not OFF_BALL_ADV_ENABLED
        elif not paused and key == ord('+'):
            fps = min(fps + 5, 60)
            frame_delay = int(1000/fps)
        elif not paused and key == ord('-'):
            fps = max(fps - 5, 1)
            frame_delay = int(1000/fps)
        elif paused and key == ord('.'):
            frame_idx += 1
        elif paused and key == ord(','):
            frame_idx -= 1
        elif not paused:
            frame_idx += 1
        
        if frame_idx < 0:
            frame_idx = 0
            
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
    ids_a_excluir = [2,23,25]
    ids_balon = [1]
    ids_equipo_atacante_muestra = [9,11,4]

    visualize_vertical_view(
        image_folder=os.path.join(base_dir, "img1"),
        detections_path=os.path.join(base_dir, "det/sorted_det.txt"),
        real_coords_path=os.path.join(base_dir, "CoordPuntosRealesColor.txt"),
        fps=10,
        output_width=600,
        exclude_ids=ids_a_excluir,
        ball_ids=ids_balon,
        attacking_team_sample_ids=ids_equipo_atacante_muestra
    )
