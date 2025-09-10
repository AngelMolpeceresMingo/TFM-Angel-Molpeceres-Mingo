import cv2
import numpy as np
import os
from collections import Counter

BASIC_COLORS_BGR = {
    "rojo": (0, 0, 255), "verde": (0, 255, 0), "azul": (255, 0, 0),
    "amarillo": (0, 255, 255), "negro": (0, 0, 0), "blanco": (255, 255, 255),
    "naranja": (0, 165, 255), "morado": (128, 0, 128)
}
BASIC_COLORS_LIST = np.array(list(BASIC_COLORS_BGR.values()))

BALL_COLOR = (255, 0, 255) 
BALL_MAX_AREA_THRESHOLD = 20 * 20 

def find_closest_basic_color(color_bgr):
    distances = np.sqrt(np.sum((BASIC_COLORS_LIST - color_bgr) ** 2, axis=1))
    return tuple(BASIC_COLORS_LIST[np.argmin(distances)])

def get_torso_dominant_basic_color(image, bbox):
    x, y, w, h = [int(val) for val in bbox]
    x1, y1, x2, y2 = max(0, x), max(0, y), min(image.shape[1], x + w), min(image.shape[0], y + h)
    if x2 <= x1 or y2 <= y1: return None

    player_crop = image[y1:y2, x1:x2]
    
    torso_h_start, torso_h_end = int(h * 0.25), int(h * 0.75)
    torso_crop = player_crop[torso_h_start:torso_h_end, :]
    if torso_crop.size == 0: return None

    hsv_crop = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2HSV)
    lower_green, upper_green = np.array([35, 40, 40]), np.array([85, 255, 255])
    mask_non_green = cv2.bitwise_not(cv2.inRange(hsv_crop, lower_green, upper_green))
    
    basic_color_pixels = [find_closest_basic_color(p) for r_idx, r in enumerate(torso_crop) for c_idx, p in enumerate(r) if mask_non_green[r_idx, c_idx]]
    
    if not basic_color_pixels: return None
    return Counter(basic_color_pixels).most_common(1)[0][0]

def load_keypoints(keypoints_path):
    keypoints_data = {}
    if not os.path.exists(keypoints_path): return keypoints_data
    with open(keypoints_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ')
            if len(parts) >= 4:
                try:
                    frame_id, point_type, x, y = int(parts[0]), ' '.join(parts[1:-2]), float(parts[-2]), float(parts[-1])
                    if frame_id not in keypoints_data: keypoints_data[frame_id] = []
                    keypoints_data[frame_id].append({'type': point_type, 'x': x, 'y': y})
                except (ValueError, IndexError): continue
    return keypoints_data

def load_detections(det_path):
    detections = {}
    if not os.path.exists(det_path): return detections
    with open(det_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 6:
                try:
                    frame_id, player_id = int(parts[0]), int(parts[1])
                    bbox_x, bbox_y, bbox_w, bbox_h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                    if frame_id not in detections: detections[frame_id] = []
                    detections[frame_id].append({'player_id': player_id, 'bbox_x': bbox_x, 'bbox_y': bbox_y, 'bbox_w': bbox_w, 'bbox_h': bbox_h})
                except ValueError: continue
    return detections

def calculate_real_coordinates_advanced(det_txt_path, keypoints_path, image_folder_path, output_real_coords_path):
    PUNTOS_REALES = {
        "Esquina Izq-Inf": (0, 0), "Esquina Izq-Sup": (0, 68), "Esquina Der-Inf": (105, 0), "Esquina Der-Sup": (105, 68),
        "18m Izq-Inf Inter": (16.5, 13.85), "18m Izq-Sup Inter": (16.5, 54.15),
        "6m Izq-Inf Inter": (5.5, 24.84), "6m Izq-Sup Inter": (5.5, 43.16),
        "18m Der-Inf Inter": (88.5, 13.85), "18m Der-Sup Inter": (88.5, 54.15),
        "6m Der-Inf Inter": (99.5, 24.84), "6m Der-Sup Inter": (99.5, 43.16),
        "Centro Campo": (52.5, 34), "Centro-Banda Sup": (52.5, 68), "Centro-Banda Inf": (52.5, 0),
        "Arco Izq Cuspide": (16.5, 34), "Arco Der Cuspide": (88.5, 34),
        "Circulo Sup": (52.5, 43.15), "Circulo Inf": (52.5, 24.85),
        "Circulo Der": (61.65, 34), "Circulo Izq": (43.35, 34),
    }

    print("Cargando datos...")
    keypoints_data = load_keypoints(keypoints_path)
    detections_data = load_detections(det_txt_path)
    if not detections_data: print("No se encontraron detecciones."); return

    print("Determinando los 3 colores de grupo principales...")
    group_colors = None
    for frame_id in sorted(detections_data.keys()):
        if len(detections_data[frame_id]) > 5:
            initial_frame_img = next((cv2.imread(p) for fn in [f"{frame_id}.jpg", f"{frame_id:06d}.jpg"] if os.path.exists(p := os.path.join(image_folder_path, fn))), None)
            if initial_frame_img is not None:
                all_player_colors = [c for p in detections_data[frame_id] if (p['bbox_w'] * p['bbox_h'] >= BALL_MAX_AREA_THRESHOLD) and (c := get_torso_dominant_basic_color(initial_frame_img, (p['bbox_x'], p['bbox_y'], p['bbox_w'], p['bbox_h']))) is not None]
                if not all_player_colors: continue
                top_3_colors = [color for color, count in Counter(all_player_colors).most_common(3)]
                if len(top_3_colors) == 3:
                    group_colors = top_3_colors
                    print(f"Colores de grupo identificados (BGR): {group_colors}")
                    break
    if group_colors is None: print("Error fatal: No se pudieron determinar los 3 colores de grupo."); return
    
    group_colors_np = np.array(group_colors)

    print("Procesando todos los frames...")
    results, last_known_homography = [], None

    for frame_id in sorted(detections_data.keys()):
        current_frame = next((cv2.imread(p) for fn in [f"{frame_id}.jpg", f"{frame_id:06d}.jpg"] if os.path.exists(p := os.path.join(image_folder_path, fn))), None)
        if current_frame is None: continue

        puntos_frame_actual = keypoints_data.get(frame_id, [])
        puntos_imagen, puntos_reales = [], []
        for kp in puntos_frame_actual:
            if kp['type'] in PUNTOS_REALES:
                puntos_imagen.append((kp['x'], kp['y'])); puntos_reales.append(PUNTOS_REALES[kp['type']])
        
        transform_matrix = None
        if len(puntos_imagen) >= 4:
            transform_matrix, _ = cv2.findHomography(np.array(puntos_imagen, dtype=np.float32), np.array(puntos_reales, dtype=np.float32), cv2.RANSAC, 5.0)
            if transform_matrix is not None: last_known_homography = transform_matrix
        
        if transform_matrix is None:
            if last_known_homography is not None:
                transform_matrix = last_known_homography
            else:
                print(f"Advertencia: Saltando frame {frame_id} porque no hay ninguna homografía previa disponible.")
                continue

        for player in detections_data[frame_id]:
            pos_img = np.array([[[player['bbox_x'] + player['bbox_w']/2, player['bbox_y'] + player['bbox_h']]]], dtype=np.float32)
            pos_real_projected = cv2.perspectiveTransform(pos_img, transform_matrix)
            x_real, y_real = pos_real_projected[0][0] if pos_real_projected is not None else (0.0, 0.0)

            if player['bbox_w'] * player['bbox_h'] < BALL_MAX_AREA_THRESHOLD:
                assigned_color = BALL_COLOR
            else:
                bbox = (player['bbox_x'], player['bbox_y'], player['bbox_w'], player['bbox_h'])
                player_basic_color = get_torso_dominant_basic_color(current_frame, bbox)
                
                if player_basic_color is not None:
                    distances = np.sqrt(np.sum((group_colors_np - player_basic_color) ** 2, axis=1))
                    assigned_color = tuple(group_colors_np[np.argmin(distances)])
                else:
                    assigned_color = (128, 128, 128)
            
            b, g, r = assigned_color
            color_str = f"{b}-{g}-{r}"
            results.append(f"{frame_id},{player['player_id']},{x_real:.2f},{y_real:.2f},{color_str}")

    print(f"Guardando resultados en {output_real_coords_path}...")
    with open(output_real_coords_path, 'w', encoding='utf-8') as f_out: f_out.write("\n".join(results))
    print("¡Proceso completado con éxito!")


if __name__ == '__main__':
    
    base_dir = "Ruta/A/La/Carpeta/De/La/Secuencia"
    path_detecciones = os.path.join(base_dir, "det/det.txt")
    path_puntos_clave = os.path.join(base_dir, "CoordPuntos.txt")
    path_imagenes = os.path.join(base_dir, "img1")
    path_salida_coords_reales = os.path.join(base_dir, "CoordPuntosRealesColor.txt")

    calculate_real_coordinates_advanced(
        det_txt_path=path_detecciones,
        keypoints_path=path_puntos_clave,
        image_folder_path=path_imagenes,
        output_real_coords_path=path_salida_coords_reales
    )
