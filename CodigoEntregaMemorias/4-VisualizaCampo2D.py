import cv2
import numpy as np
import os

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
    
def visualize_vertical_view(image_folder, detections_path, real_coords_path, fps=10, output_width=1280):
    print("Cargando todos los datos...")
    image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if not image_files:
        print(f"No se encontraron imágenes en {image_folder}"); return

    PUNTOS_REALES = {
        "Esquina Izq-Inf": (0, 0), "Esquina Izq-Sup": (0, 68), "Esquina Der-Inf": (105, 0), "Esquina Der-Sup": (105, 68),
        "18m Izq-Inf Inter": (16.5, 13.85), "18m Izq-Sup Inter": (16.5, 54.15), "6m Izq-Inf Inter": (5.5, 24.84), "6m Izq-Sup Inter": (5.5, 43.16),
        "18m Der-Inf Inter": (88.5, 13.85), "18m Der-Sup Inter": (88.5, 54.15), "6m Der-Inf Inter": (99.5, 24.84), "6m Der-Sup Inter": (99.5, 43.16),
        "Centro Campo": (52.5, 34), "Centro-Banda Sup": (52.5, 68), "Centro-Banda Inf": (52.5, 0), "Arco Izq Cuspide": (16.5, 34), "Arco Der Cuspide": (88.5, 34),
        "Circulo Sup": (52.5, 34 + 9.15), "Circulo Inf": (52.5, 34 - 9.15), "Circulo Der": (52.5 + 9.15, 34), "Circulo Izq": (52.5 - 9.15, 34),
    }

    detections_data = load_det_data(detections_path)

    real_coords_data = load_real_coords_with_color(real_coords_path)
    
    field_template = draw_soccer_field(PUNTOS_REALES, output_width=output_width)
    paused, frame_delay = False, int(1000 / fps)
    
    print("\n=== CONTROLES DE VISUALIZACIÓN ===")
    print("'q': Salir, ESPACIO: Pausar/Reanudar, +/-: Velocidad")
    
    for image_file in image_files:
        frame_id = int(os.path.splitext(image_file)[0])
        original_frame = cv2.imread(os.path.join(image_folder, image_file))
        if original_frame is None: continue

        if frame_detections := detections_data.get(frame_id):
            for det in frame_detections:
                display_text = str(det['player_id'])
                cv2.rectangle(original_frame, (int(det['bbox_x']), int(det['bbox_y'])), (int(det['bbox_x'] + det['bbox_w']), int(det['bbox_y'] + det['bbox_h'])), (0, 255, 0), 2)
                cv2.putText(original_frame, display_text, (int(det['bbox_x']), int(det['bbox_y']) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        field_2d_view = field_template.copy()
        
        if frame_real_coords := real_coords_data.get(frame_id):
            for coord_info in frame_real_coords:
                field_h, field_w = field_2d_view.shape[:2]
                x_px = int(coord_info['x'] / 105.0 * field_w)
                y_px = int(coord_info['y'] / 68.0 * field_h)
                
                player_id = coord_info['player_id']
                player_color = coord_info['color']
                
                cv2.circle(field_2d_view, (x_px, y_px), 5, player_color, -1)
                cv2.circle(field_2d_view, (x_px, y_px), 5, (0,0,0), 1)
                
                if player_id != -1:
                    cv2.putText(field_2d_view, str(player_id), (x_px + 8, y_px + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        padded_frame = resize_and_pad(original_frame, target_width=output_width)
        padded_field = field_2d_view
        
        status_text = f"Frame: {frame_id} | FPS: {fps}"
        if paused: status_text += " [PAUSADO]"
        cv2.putText(padded_frame, status_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(padded_field, "Vista 2D", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        combined_view = np.vstack((padded_frame, padded_field))
        
        cv2.imshow("Vista Combinada: Partido (Arriba) vs Campo 2D (Abajo)", combined_view)

        while True:
            key = cv2.waitKey(1 if paused else frame_delay) & 0xFF
            if key == ord('q'): cv2.destroyAllWindows(); return
            elif key == ord(' '): paused = not paused; break
            elif key == ord('+'): fps = min(fps + 5, 60); frame_delay = int(1000/fps); break
            elif key == ord('-'): fps = max(fps - 5, 1); frame_delay = int(1000/fps); break
            elif not paused: break
                
    cv2.destroyAllWindows()

if __name__ == '__main__':
    base_dir = "Ruta/A/La/Carpeta/De/La/Secuencia"
    visualize_vertical_view(
        image_folder=os.path.join(base_dir, "img1"),
        detections_path=os.path.join(base_dir, "det/sorted_det.txt"),
        real_coords_path=os.path.join(base_dir, "CoordPuntosRealesColor.txt"),
        fps=5,
        output_width=600
    )
