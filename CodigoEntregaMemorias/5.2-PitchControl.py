import cv2
import numpy as np
import os
from scipy.spatial import Voronoi
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from collections import Counter

TEAM1_FIXED_COLOR_BGR = (255, 0, 0) 
TEAM2_FIXED_COLOR_BGR = (0, 0, 255) 
REFEREE_FIXED_COLOR_BGR = (0, 255, 255) 
BALL_COLOR_BGR = (128, 0, 128)
BALL_RING_COLOR_BGR = (128, 0, 128) 
TEAM1_FIXED_COLOR_RGB = (0, 0, 255)
TEAM2_FIXED_COLOR_RGB = (255, 0, 0) 
REFEREE_FIXED_COLOR_RGB = (255, 255, 0) 

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
                    frame_id = int(parts[0])
                    player_id = int(parts[1])
                    x = float(parts[2])
                    y = float(parts[3])
                    
                    color_parts = parts[4].split('-')
                    color = (int(color_parts[0]), int(color_parts[1]), int(color_parts[2]))
                    
                    if frame_id not in data:
                        data[frame_id] = []
                    
                    data[frame_id].append({
                        'player_id': player_id, 
                        'x': x, 
                        'y': y, 
                        'color': color
                    })
                except (ValueError, IndexError):
                    continue
    return data

def analyze_team_colors(real_coords_data):
    all_colors = []
    
    for frame_data in real_coords_data.values():
        for player_data in frame_data:
            all_colors.append(player_data['color'])
    
    color_counts = Counter(all_colors)
    unique_colors = list(color_counts.keys())
    
    if len(unique_colors) < 2:
        print("Advertencia: Se encontraron menos de 2 colores únicos")
        return None, None, None
    
    if len(unique_colors) == 2:
        team1_color = unique_colors[0]
        team2_color = unique_colors[1]
        referee_color = (128, 128, 128) 
        print(f"Detectados 2 colores: Equipo 1 {team1_color}, Equipo 2 {team2_color}")
        return team1_color, team2_color, referee_color
    
    colors_array = np.array(unique_colors)
    
    kmeans = KMeans(n_clusters=min(3, len(unique_colors)), random_state=42)
    cluster_labels = kmeans.fit_predict(colors_array)
    centroids = kmeans.cluster_centers_.astype(int)
    
    cluster_sizes = []
    for i in range(len(centroids)):
        cluster_colors = [unique_colors[j] for j, label in enumerate(cluster_labels) if label == i]
        total_count = sum(color_counts[color] for color in cluster_colors)
        cluster_sizes.append((i, total_count, tuple(centroids[i])))
    
    cluster_sizes.sort(key=lambda x: x[1], reverse=True)
    
    team1_color = cluster_sizes[0][2]
    team2_color = cluster_sizes[1][2] if len(cluster_sizes) > 1 else (255, 0, 0)
    referee_color = cluster_sizes[2][2] if len(cluster_sizes) > 2 else (128, 128, 128)
    
    print(f"Colores detectados - Equipo 1: {team1_color}, Equipo 2: {team2_color}, Árbitros: {referee_color}")
    return team1_color, team2_color, referee_color

def classify_player_by_color(player_color, team1_color, team2_color, referee_color):
    def color_distance(c1, c2):
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))
    
    dist_team1 = color_distance(player_color, team1_color)
    dist_team2 = color_distance(player_color, team2_color)
    dist_referee = color_distance(player_color, referee_color)
    
    min_dist = min(dist_team1, dist_team2, dist_referee)
    
    if min_dist == dist_team1:
        return 'team1'
    elif min_dist == dist_team2:
        return 'team2'
    else:
        return 'referee'

def draw_player_with_team_ring(field_2d_view, x_px, y_px, player_color_rgb, team_assignment, p_id, exclude=False, is_ball=False):
    if is_ball:
        radius = 5
        cv2.circle(field_2d_view, (x_px, y_px), radius, BALL_COLOR_BGR, -1) 
        cv2.circle(field_2d_view, (x_px, y_px), radius, BALL_RING_COLOR_BGR, 2) 
        if p_id != -1:
            cv2.putText(field_2d_view, str(p_id), (x_px + 8, y_px + 4), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1) 
        return
    
    if team_assignment == 'team1':
        ring_color_bgr = TEAM1_FIXED_COLOR_BGR
        player_color_bgr = (player_color_rgb[2], player_color_rgb[1], player_color_rgb[0])
        radius = 5
    elif team_assignment == 'team2':
        ring_color_bgr = TEAM2_FIXED_COLOR_BGR
        player_color_bgr = (player_color_rgb[2], player_color_rgb[1], player_color_rgb[0])
        radius = 5
    else:
        ring_color_bgr = REFEREE_FIXED_COLOR_BGR
        player_color_bgr = REFEREE_FIXED_COLOR_BGR
        radius = 3
    
    if exclude:
        cv2.circle(field_2d_view, (x_px, y_px), 2, player_color_bgr, -1)
        if team_assignment in ['team1', 'team2']:
            cv2.circle(field_2d_view, (x_px, y_px), 2, ring_color_bgr, 1)
    else:
        cv2.circle(field_2d_view, (x_px, y_px), radius, player_color_bgr, -1)
        if team_assignment in ['team1', 'team2']:
            cv2.circle(field_2d_view, (x_px, y_px), radius, ring_color_bgr, 2)
        
        if p_id != -1:
            offset_x = 8 if team_assignment in ['team1', 'team2'] else 5
            cv2.putText(field_2d_view, str(p_id), (x_px + offset_x, y_px + 4), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

def load_det_data(file_path):
    detections = {}
    if not os.path.exists(file_path):
        print(f"Error: No se pudo encontrar el archivo de detecciones en la ruta: {file_path}")
        return detections
    with open(file_path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            try:
                parts = line.strip().split(',')
                if len(parts) >= 6:
                    frame_id, player_id = int(parts[0]), int(parts[1])
                    if frame_id not in detections: detections[frame_id] = []
                    detections[frame_id].append({
                        'player_id': player_id, 'bbox_x': float(parts[2]), 'bbox_y': float(parts[3]),
                        'bbox_w': float(parts[4]), 'bbox_h': float(parts[5]), 'numbers': []
                    })
            except (ValueError, IndexError): continue
    return detections

def draw_soccer_field(puntos_reales_dict, output_width=1280):
    field_length_m, field_width_m = 105.0, 68.0
    aspect_ratio = field_width_m / field_length_m
    output_height = int(output_width * aspect_ratio)
    field = np.full((output_height, output_width, 3), (20, 110, 25), dtype=np.uint8)
    line_color, line_thickness = (255, 255, 255), 2
    
    def scale(coords):
        x_m, y_m = coords
        x_px = int(x_m / field_length_m * output_width)
        y_px = int(y_m / field_width_m * output_height)
        return x_px, y_px
    
    cv2.rectangle(field, scale(puntos_reales_dict["Esquina Izq-Inf"]), scale(puntos_reales_dict["Esquina Der-Sup"]), line_color, line_thickness)
    cv2.line(field, scale(puntos_reales_dict["Centro-Banda Inf"]), scale(puntos_reales_dict["Centro-Banda Sup"]), line_color, line_thickness)
    radius_m = 9.15
    radius_px = int(radius_m / field_length_m * output_width)
    cv2.circle(field, scale(puntos_reales_dict["Centro Campo"]), radius_px, line_color, line_thickness)
    
    cv2.rectangle(field, scale(puntos_reales_dict["18m Izq-Inf Inter"]), scale((0, puntos_reales_dict["18m Izq-Sup Inter"][1])), line_color, line_thickness)
    cv2.rectangle(field, scale(puntos_reales_dict["18m Der-Inf Inter"]), scale((105, puntos_reales_dict["18m Der-Sup Inter"][1])), line_color, line_thickness)
    cv2.rectangle(field, scale(puntos_reales_dict["6m Izq-Inf Inter"]), scale((0, puntos_reales_dict["6m Izq-Sup Inter"][1])), line_color, line_thickness)
    cv2.rectangle(field, scale(puntos_reales_dict["6m Der-Inf Inter"]), scale((105, puntos_reales_dict["6m Der-Sup Inter"][1])), line_color, line_thickness)
    penalty_spot_izq, penalty_spot_der = (11, 34), (94, 34)
    cv2.ellipse(field, scale(penalty_spot_izq), (radius_px, radius_px), 0, -53, 53, line_color, line_thickness)
    cv2.ellipse(field, scale(penalty_spot_der), (radius_px, radius_px), 0, 127, 233, line_color, line_thickness)
            
    return field

def resize_and_pad(image, target_width):
    original_h, original_w = image.shape[:2]
    if original_w == target_width: return image
    scale = target_width / original_w
    return cv2.resize(image, (target_width, int(original_h * scale)))

INFLUENCE_STD_DEV = 10.0 
GRID_RESOLUTION = 20 

def player_influence(field_point, player_position, std_dev=INFLUENCE_STD_DEV):
    mean = np.array(player_position)
    cov = np.array([[std_dev**2, 0], [0, std_dev**2]])
    if std_dev == 0: return 1.0 if np.allclose(field_point, player_position) else 0.0
    density = multivariate_normal.pdf(field_point, mean=mean, cov=cov)
    max_pdf = 1 / (2 * np.pi * std_dev**2)
    normalized_influence = density / max_pdf
    return normalized_influence

def calculate_pitch_control_map(players_team1, players_team2, field_dims_m, output_dims_px):

    field_length_m, field_width_m = field_dims_m
    field_height_px, field_width_px = output_dims_px
    x_coords_m = np.linspace(0, field_length_m, GRID_RESOLUTION)
    y_coords_m = np.linspace(0, field_width_m, GRID_RESOLUTION)
    pitch_control_values = np.zeros((GRID_RESOLUTION, GRID_RESOLUTION))

    for i, x_m in enumerate(x_coords_m):
        for j, y_m in enumerate(y_coords_m):
            field_point = np.array([x_m, y_m])
            sum_influence_team1 = sum(player_influence(field_point, [p['x'], p['y']]) for p in players_team1)
            sum_influence_team2 = sum(player_influence(field_point, [p['x'], p['y']]) for p in players_team2)
            diff_influence = sum_influence_team1 - sum_influence_team2
            pitch_control_values[j, i] = 1 / (1 + np.exp(-diff_influence))

    resized_pc_map = cv2.resize(pitch_control_values, (field_width_px, field_height_px), interpolation=cv2.INTER_LINEAR)

    heatmap = np.zeros((field_height_px, field_width_px, 3), dtype=np.uint8)
    
    neutral_intensity = 180  
    color_intensity = 1.3 
    
    mask_team1 = resized_pc_map > 0.5
    ratio_team1 = ((resized_pc_map[mask_team1] - 0.5) / 0.5) * color_intensity
    ratio_team1 = np.clip(ratio_team1, 0, 1) 
    
    heatmap[mask_team1, 0] = TEAM1_FIXED_COLOR_BGR[0] * ratio_team1 + neutral_intensity * (1 - ratio_team1)
    heatmap[mask_team1, 1] = TEAM1_FIXED_COLOR_BGR[1] * ratio_team1 + neutral_intensity * (1 - ratio_team1)
    heatmap[mask_team1, 2] = TEAM1_FIXED_COLOR_BGR[2] * ratio_team1 + neutral_intensity * (1 - ratio_team1)
    
    mask_team2 = resized_pc_map < 0.5
    ratio_team2 = (1 - (resized_pc_map[mask_team2] / 0.5)) * color_intensity
    ratio_team2 = np.clip(ratio_team2, 0, 1)
    
    heatmap[mask_team2, 0] = TEAM2_FIXED_COLOR_BGR[0] * ratio_team2 + neutral_intensity * (1 - ratio_team2)
    heatmap[mask_team2, 1] = TEAM2_FIXED_COLOR_BGR[1] * ratio_team2 + neutral_intensity * (1 - ratio_team2)
    heatmap[mask_team2, 2] = TEAM2_FIXED_COLOR_BGR[2] * ratio_team2 + neutral_intensity * (1 - ratio_team2)
    
    heatmap[resized_pc_map == 0.5] = [neutral_intensity, neutral_intensity, neutral_intensity]
    
    return heatmap

def visualize_vertical_view(image_folder, detections_path, real_coords_path, fps=10, output_width=1280, exclude_ids=None, show_pitch_control=True, ball_id=None):
    if exclude_ids is None:
        exclude_ids = []

    print("Cargando todos los datos...")
    image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if not image_files:
        print(f"No se encontraron imágenes en {image_folder}"); return

    PUNTOS_REALES = {
        "Esquina Izq-Inf": (0, 0), "Esquina Izq-Sup": (0, 68), "Esquina Der-Inf": (105, 0), "Esquina Der-Sup": (105, 68),
        "18m Izq-Inf Inter": (16.5, 13.85), "18m Izq-Sup Inter": (16.5, 54.15), "6m Izq-Inf Inter": (5.5, 24.84), "6m Izq-Sup Inter": (5.5, 43.16),
        "18m Der-Inf Inter": (88.5, 13.85), "18m Der-Sup Inter": (88.5, 54.15), "6m Der-Inf Inter": (99.5, 24.84), "6m Der-Sup Inter": (99.5, 43.16),
        "Centro Campo": (52.5, 34), "Centro-Banda Sup": (52.5, 68), "Centro-Banda Inf": (52.5, 0),
    }

    detections_data = load_det_data(detections_path)
    real_coords_data = load_real_coords_with_color(real_coords_path)
    
    team1_color, team2_color, referee_color = analyze_team_colors(real_coords_data)
    if team1_color is None:
        print("Error: No se pudieron detectar los colores de los equipos")
        return
    
    field_template = draw_soccer_field(PUNTOS_REALES, output_width=output_width)
    field_length_m, field_width_m = 105.0, 68.0
    field_height_px, field_width_px = field_template.shape[:2]
    overlay_alpha, paused, frame_delay = 0.5, False, int(1000 / fps)
    
    print("\n=== CONTROLES DE VISUALIZACIÓN ===")
    print("'q': Salir, ESPACIO: Pausar/Reanudar, +/-: Velocidad, 'm': Cambiar Vista (Voronoi/Pitch Control)")
    if ball_id:
        print(f"Balón detectado con ID: {ball_id}")

    for image_file in image_files:
        frame_id = int(os.path.splitext(image_file)[0])
        original_frame = cv2.imread(os.path.join(image_folder, image_file))
        if original_frame is None: continue

        padded_frame = resize_and_pad(original_frame, target_width=output_width)
        original_h, original_w = original_frame.shape[:2]
        scale_factor = padded_frame.shape[1] / original_w

        if frame_detections := detections_data.get(frame_id):
            for det in frame_detections:
                bbox_x_s = int(det['bbox_x'] * scale_factor)
                bbox_y_s = int(det['bbox_y'] * scale_factor)
                bbox_w_s = int(det['bbox_w'] * scale_factor)
                bbox_h_s = int(det['bbox_h'] * scale_factor)
                
                display_text = str(det['player_id'])
                cv2.rectangle(padded_frame, (bbox_x_s, bbox_y_s), (bbox_x_s + bbox_w_s, bbox_y_s + bbox_h_s), (0, 255, 0), 2)
                cv2.putText(padded_frame, display_text, (bbox_x_s, bbox_y_s - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        status_text = f"Frame: {frame_id} | FPS: {fps} | Vista: {'Pitch Control' if show_pitch_control else 'Voronoi'}"
        if paused: status_text += " [PAUSADO]"
        cv2.putText(padded_frame, status_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        field_2d_view = field_template.copy()
        
        if frame_real_coords := real_coords_data.get(frame_id):
            players_team1, players_team2 = [], []
            voronoi_player_positions_m, voronoi_player_colors = [], []
            
            for coord_info in frame_real_coords:
                if coord_info['player_id'] not in exclude_ids:
                    is_ball = (ball_id is not None and coord_info['player_id'] == ball_id)
                    
                    if not is_ball:
                        team_assignment = classify_player_by_color(
                            coord_info['color'], team1_color, team2_color, referee_color
                        )
                        
                        if team_assignment == 'team1':
                            players_team1.append({'x': coord_info['x'], 'y': coord_info['y']})
                        elif team_assignment == 'team2':
                            players_team2.append({'x': coord_info['x'], 'y': coord_info['y']})
                    
                    voronoi_player_positions_m.append([coord_info['x'], coord_info['y']])
                    voronoi_player_colors.append(coord_info['color'])

            overlay_image = None
            if show_pitch_control:
                if players_team1 or players_team2:
                    overlay_image = calculate_pitch_control_map(
                        players_team1, players_team2, 
                        (field_length_m, field_width_m), 
                        (field_height_px, field_width_px)
                    )
                cv2.putText(field_2d_view, "Vista 2D (Pitch Control)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            else:
                if len(voronoi_player_positions_m) >= 4:
                    points_px = np.array([[int(x/field_length_m*field_width_px), int(y/field_width_m*field_height_px)] for x, y in voronoi_player_positions_m])
                    padding = max(field_width_px, field_height_px)
                    bbox_points = np.array([[-padding, -padding], [field_width_px+padding, -padding], [-padding, field_height_px+padding], [field_width_px+padding, field_height_px+padding]])
                    all_points = np.vstack([points_px, bbox_points])
                    vor = Voronoi(all_points)
                    voronoi_regions_img = np.zeros_like(field_2d_view)
                    
                    for i, region_idx in enumerate(vor.point_region):
                        if i < len(points_px):
                            region = vor.regions[region_idx]
                            if not -1 in region:
                                polygon = np.array([vor.vertices[j] for j in region], dtype=np.int32)
                                cv2.fillPoly(voronoi_regions_img, [polygon], voronoi_player_colors[i])
                    overlay_image = voronoi_regions_img
                cv2.putText(field_2d_view, "Vista 2D (Voronoi)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            if overlay_image is not None:
                field_2d_view = cv2.addWeighted(field_2d_view, 1, overlay_image, overlay_alpha, 0)
            
            for coord_info in frame_real_coords:
                x_px, y_px = int(coord_info['x']/field_length_m*field_width_px), int(coord_info['y']/field_width_m*field_height_px)
                p_id = coord_info['player_id']
                
                is_ball = (ball_id is not None and p_id == ball_id)
                
                if not is_ball:
                    team_assignment = classify_player_by_color(
                        coord_info['color'], team1_color, team2_color, referee_color
                    )
                else:
                    team_assignment = 'ball'
                
                draw_player_with_team_ring(
                    field_2d_view, x_px, y_px, 
                    coord_info['color'],
                    team_assignment,
                    p_id,
                    exclude=(p_id in exclude_ids),
                    is_ball=is_ball
                )
        
        combined_view = np.vstack((padded_frame, field_2d_view))
        cv2.imshow("Vista Combinada: Partido (Arriba) vs Campo 2D (Abajo)", combined_view)

        while True:
            key = cv2.waitKey(1 if paused else frame_delay) & 0xFF
            if key == ord('q'): cv2.destroyAllWindows(); return
            elif key == ord(' '): paused = not paused; break
            elif key == ord('+'): fps = min(fps + 5, 60); frame_delay = int(1000/fps); break
            elif key == ord('-'): fps = max(fps - 5, 1); frame_delay = int(1000/fps); break
            elif key == ord('m'): show_pitch_control = not show_pitch_control; break
            elif not paused: break
                
    cv2.destroyAllWindows()

if __name__ == '__main__':
    base_dir = "Ruta/A/La/Carpeta/De/La/Secuencia"
    ids_a_excluir = [2,15, 6]
    id_balon = 3 

    visualize_vertical_view(
        image_folder=os.path.join(base_dir, "img1"),
        detections_path=os.path.join(base_dir, "det/det.txt"),
        real_coords_path=os.path.join(base_dir, "CoordPuntosRealesColor.txt"),
        fps=5,
        output_width=600,
        exclude_ids=ids_a_excluir,
        show_pitch_control=True,
        ball_id=id_balon
    )
