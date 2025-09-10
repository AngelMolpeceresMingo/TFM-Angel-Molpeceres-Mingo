import cv2
import numpy as np
import yaml
import torch
import argparse
import os
import glob
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as f

from model.cls_hrnet import get_cls_net
from model.cls_hrnet_l import get_cls_net as get_cls_net_l
from utils.utils_calib import FramebyFrameCalib
from utils.utils_heatmap import get_keypoints_from_heatmap_batch_maxpool, get_keypoints_from_heatmap_batch_maxpool_l, complete_keypoints, coords_to_dict

lines_coords = [[[0., 54.16, 0.], [16.5, 54.16, 0.]],
                [[16.5, 13.84, 0.], [16.5, 54.16, 0.]],
                [[16.5, 13.84, 0.], [0., 13.84, 0.]],
                [[88.5, 54.16, 0.], [105., 54.16, 0.]],
                [[88.5, 13.84, 0.], [88.5, 54.16, 0.]],
                [[88.5, 13.84, 0.], [105., 13.84, 0.]],
                [[0., 37.66, -2.44], [0., 30.34, -2.44]],
                [[0., 37.66, 0.], [0., 37.66, -2.44]],
                [[0., 30.34, 0.], [0., 30.34, -2.44]],
                [[105., 37.66, -2.44], [105., 30.34, -2.44]],
                [[105., 30.34, 0.], [105., 30.34, -2.44]],
                [[105., 37.66, 0.], [105., 37.66, -2.44]],
                [[52.5, 0., 0.], [52.5, 68, 0.]],
                [[0., 68., 0.], [105., 68., 0.]],
                [[0., 0., 0.], [0., 68., 0.]],
                [[105., 0., 0.], [105., 68., 0.]],
                [[0., 0., 0.], [105., 0., 0.]],
                [[0., 43.16, 0.], [5.5, 43.16, 0.]],
                [[5.5, 43.16, 0.], [5.5, 24.84, 0.]],
                [[5.5, 24.84, 0.], [0., 24.84, 0.]],
                [[99.5, 43.16, 0.], [105., 43.16, 0.]],
                [[99.5, 43.16, 0.], [99.5, 24.84, 0.]],
                [[99.5, 24.84, 0.], [105., 24.84, 0.]]]

def load_detections(detection_file):

    detections = {}
    with open(detection_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            frame_id = int(parts[0])
            player_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            confidence = float(parts[6])
            
            numbers_str = parts[10].strip("[]'")
            numbers = [n.strip().strip("'") for n in numbers_str.split(',') if n.strip().strip("'")]
            
            if frame_id not in detections:
                detections[frame_id] = []
            
            detections[frame_id].append({
                'player_id': player_id,
                'bbox': [x, y, w, h],
                'confidence': confidence,
                'numbers': numbers,
                'foot_pos': [x + w/2, y + h]
            })
    
    return detections

def pixel_to_field_coordinates(pixel_x, pixel_y, P, field_height=0.0):
    
    try:
        P_inv = np.linalg.pinv(P)
        pixel_coords = np.array([pixel_x, pixel_y, 1.0])
        world_coords_4d = P_inv @ pixel_coords
        
        if world_coords_4d[3] != 0:
            world_coords_3d = world_coords_4d[:3] / world_coords_4d[3]
        else:
            return None, None
        
        field_x = world_coords_3d[0] + 105/2
        field_y = world_coords_3d[1] + 68/2
        
        return field_x, field_y
    except:
        return None, None

def draw_field_minimap(frame, field_positions, frame_width=300, frame_height=200):
    
    minimap = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    minimap.fill(50)
    
    field_length = 105.0
    field_width = 68.0
    
    scale_x = (frame_width - 40) / field_length
    scale_y = (frame_height - 40) / field_width
    
    cv2.rectangle(minimap, (20, 20), (frame_width-20, frame_height-20), (0, 150, 0), 2)
    
    center_x = int(20 + field_length/2 * scale_x)
    cv2.line(minimap, (center_x, 20), (center_x, frame_height-20), (255, 255, 255), 1)
    cv2.circle(minimap, (center_x, frame_height//2), int(9.15 * scale_x), (255, 255, 255), 1)
    
    area_left_x = int(20 + 16.5 * scale_x)
    area_top = int(20 + (field_width/2 - 20.15) * scale_y)
    area_bottom = int(20 + (field_width/2 + 20.15) * scale_y)
    cv2.rectangle(minimap, (20, area_top), (area_left_x, area_bottom), (255, 255, 255), 1)
    
    area_right_x = int(20 + (field_length - 16.5) * scale_x)
    cv2.rectangle(minimap, (area_right_x, area_top), (frame_width-20, area_bottom), (255, 255, 255), 1)
    
    for pos in field_positions:
        if pos['field_x'] is not None and pos['field_y'] is not None:
            map_x = int(20 + pos['field_x'] * scale_x)
            map_y = int(20 + pos['field_y'] * scale_y)
            
            if 20 <= map_x <= frame_width-20 and 20 <= map_y <= frame_height-20:
                if pos['numbers']:
                    color = (0, 255, 255)
                else:
                    color = (255, 0, 0)
                
                cv2.circle(minimap, (map_x, map_y), 4, color, -1)
                cv2.circle(minimap, (map_x, map_y), 4, (255, 255, 255), 1)
                
                if pos['numbers']:
                    number_text = pos['numbers'][0]
                    cv2.putText(minimap, number_text, (map_x-5, map_y-8), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    return minimap

def projection_from_cam_params(final_params_dict):
    
    cam_params = final_params_dict["cam_params"]
    x_focal_length = cam_params['x_focal_length']
    y_focal_length = cam_params['y_focal_length']
    principal_point = np.array(cam_params['principal_point'])
    position_meters = np.array(cam_params['position_meters'])
    rotation = np.array(cam_params['rotation_matrix'])

    It = np.eye(4)[:-1]
    It[:, -1] = -position_meters
    Q = np.array([[x_focal_length, 0, principal_point[0]],
                  [0, y_focal_length, principal_point[1]],
                  [0, 0, 1]])
    P = Q @ (rotation @ It)
    return P

def inference(cam, frame, model, model_l, kp_threshold, line_threshold, pnl_refine):
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    frame_tensor = f.to_tensor(frame_pil).float().unsqueeze(0)
    
    _, _, h_original, w_original = frame_tensor.size()
    if frame_tensor.size()[-1] != 960:
        transform2 = T.Resize((540, 960))
        frame_tensor = transform2(frame_tensor)
    
    frame_tensor = frame_tensor.to(device)
    b, c, h, w = frame_tensor.size()

    with torch.no_grad():
        heatmaps = model(frame_tensor)
        heatmaps_l = model_l(frame_tensor)

    kp_coords = get_keypoints_from_heatmap_batch_maxpool(heatmaps[:,:-1,:,:])
    line_coords = get_keypoints_from_heatmap_batch_maxpool_l(heatmaps_l[:,:-1,:,:])
    kp_dict = coords_to_dict(kp_coords, threshold=kp_threshold)
    lines_dict = coords_to_dict(line_coords, threshold=line_threshold)
    kp_dict, lines_dict = complete_keypoints(kp_dict[0], lines_dict[0], w=w, h=h, normalize=True)

    cam.update(kp_dict, lines_dict)
    final_params_dict = cam.heuristic_voting(refine_lines=pnl_refine)
    return final_params_dict

def process_images_with_players(input_folder, detection_file, model_kp, model_line, 
                               kp_threshold, line_threshold, pnl_refine, save_path, display):
    
    print(" Cargando archivo de detecciones...")
    detections = load_detections(detection_file)
    total_detections = sum(len(dets) for dets in detections.values())
    print(f" Cargadas {total_detections} detecciones en {len(detections)} frames")

    print(" Escaneando imágenes JPG...")
    image_files = sorted(glob.glob(os.path.join(input_folder, "*.jpg")))
    if not image_files:
        print(f" ERROR: No se encontraron imágenes JPG en {input_folder}")
        return
    
    print(f" Encontradas {len(image_files)} imágenes JPG")
    
    print(" Configurando procesamiento...")
    first_frame = cv2.imread(image_files[0])
    if first_frame is None:
        print(f" ERROR: No se pudo leer la primera imagen {image_files[0]}")
        return
        
    frame_height, frame_width = first_frame.shape[:2]
    print(f" Resolución: {frame_width}x{frame_height}")
    
    cam = FramebyFrameCalib(iwidth=frame_width, iheight=frame_height, denormalize=True)
    
    positions_file = 'player_positions.txt'
    print(f" Archivo de posiciones: {positions_file}")

    print("\n Iniciando procesamiento de frames...")
    pbar = tqdm(
        total=len(image_files), 
        desc=" Localizando jugadores",
        unit="frame",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
    )
    
    frames_con_calibracion = 0
    frames_con_jugadores = 0
    total_jugadores_procesados = 0
    frames_con_errores = 0
    
    with open(positions_file, 'w') as pos_file:
        pos_file.write("frame_id,player_id,pixel_x,pixel_y,field_x,field_y,confidence,numbers\n")
        
        frame_count = 1
        
        for image_file in image_files:
            frame_name = os.path.basename(image_file)
            
            frame = cv2.imread(image_file)
            if frame is None:
                frames_con_errores += 1
                pbar.set_postfix({
                    "Frame": frame_name,
                    "Estado": " Error lectura",
                    "Calib": f"{frames_con_calibracion}/{frame_count}",
                    "Jugadores": total_jugadores_procesados
                })
                pbar.update(1)
                frame_count += 1
                continue
            
            final_params_dict = inference(cam, frame, model_kp, model_l, kp_threshold, line_threshold, pnl_refine)
            
            jugadores_en_frame = 0
            estado_calibracion = " Sin calib"
            
            if final_params_dict is not None:
                frames_con_calibracion += 1
                estado_calibracion = " Calibrado"
                
                P = projection_from_cam_params(final_params_dict)
                
                frame_detections = detections.get(frame_count, [])
                field_positions = []
                jugadores_en_frame = len(frame_detections)
                
                if jugadores_en_frame > 0:
                    frames_con_jugadores += 1
                    total_jugadores_procesados += jugadores_en_frame
                
                for detection in frame_detections:
                    pixel_x, pixel_y = detection['foot_pos']
                    
                    field_x, field_y = pixel_to_field_coordinates(pixel_x, pixel_y, P)
                    
                    field_positions.append({
                        'player_id': detection['player_id'],
                        'pixel_x': pixel_x,
                        'pixel_y': pixel_y,
                        'field_x': field_x,
                        'field_y': field_y,
                        'confidence': detection['confidence'],
                        'numbers': detection['numbers']
                    })
                    
                    numbers_str = '|'.join(detection['numbers']) if detection['numbers'] else ''
                    pos_file.write(f"{frame_count},{detection['player_id']},{pixel_x:.2f},{pixel_y:.2f},"
                                 f"{field_x:.2f},{field_y:.2f},{detection['confidence']:.3f},{numbers_str}\n")
                
                    if field_x is not None and field_y is not None:
                        x, y, w, h = detection['bbox']
                        cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
                        
                        cv2.circle(frame, (int(pixel_x), int(pixel_y)), 5, (0, 0, 255), -1)
                        
                        info_text = f"ID:{detection['player_id']}"
                        if detection['numbers']:
                            info_text += f" #{detection['numbers'][0]}"
                        info_text += f" ({field_x:.1f},{field_y:.1f})"
                        
                        cv2.putText(frame, info_text, (int(x), int(y)-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        cv2.putText(frame, info_text, (int(x), int(y)-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                minimap = draw_field_minimap(frame, field_positions)
                
                y_offset = 10
                x_offset = frame_width - minimap.shape[1] - 10
                frame[y_offset:y_offset+minimap.shape[0], 
                      x_offset:x_offset+minimap.shape[1]] = minimap
                
                info_text = f"Frame: {frame_count} | Jugadores: {len(frame_detections)}"
                cv2.putText(frame, info_text, (10, frame_height-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, info_text, (10, frame_height-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            
            if save_path and out is not None:
                out.write(frame)
            
            if display:
                cv2.imshow('Player Localization', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n Procesamiento interrumpido por el usuario")
                    break
            
            progreso_pct = (frame_count / len(image_files)) * 100
            pbar.set_postfix({
                "Frame": frame_name,
                "Estado": estado_calibracion,
                "Jugadores": f"{jugadores_en_frame}J",
                "Calib": f"{frames_con_calibracion}/{frame_count}",
                "Total J": total_jugadores_procesados,
                "Progreso": f"{progreso_pct:.1f}%"
            })
            
            pbar.update(1)
            frame_count += 1
    
    pbar.close()
    
    if save_path and out is not None:
        out.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Localizar jugadores en secuencias de imágenes usando PnLCalib")
    parser.add_argument("--weights_kp", type=str, required=True, help="Modelo para keypoints")
    parser.add_argument("--weights_line", type=str, required=True, help="Modelo para líneas")
    parser.add_argument("--kp_threshold", type=float, default=0.3434, help="Umbral keypoints")
    parser.add_argument("--line_threshold", type=float, default=0.7867, help="Umbral líneas")
    parser.add_argument("--pnl_refine", action="store_true", help="Activar refinamiento PnL")
    parser.add_argument("--device", type=str, default="cpu", help="CPU o CUDA")
    parser.add_argument("--input_folder", type=str, required=True, help="Carpeta con imágenes JPG")
    parser.add_argument("--detection_file", type=str, required=True, help="Archivo Newdet.txt")
    parser.add_argument("--save_path", type=str, default="", help="Video de salida (opcional)")
    parser.add_argument("--display", action="store_true", help="Mostrar en tiempo real")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path_kp = os.path.join(script_dir, "config", "hrnetv2_w48.yaml")
    config_path_line = os.path.join(script_dir, "config", "hrnetv2_w48_l.yaml")
    weights_path_kp = os.path.join(script_dir, "weights", args.weights_kp)
    weights_path_line = os.path.join(script_dir, "weights", args.weights_line)
    
    cfg = yaml.safe_load(open(config_path_kp, 'r'))
    cfg_l = yaml.safe_load(open(config_path_line, 'r'))
    
    model = get_cls_net(cfg)
    model.load_state_dict(torch.load(weights_path_kp, map_location=device))
    model.to(device)
    model.eval()
    
    model_l = get_cls_net_l(cfg_l)
    model_l.load_state_dict(torch.load(weights_path_line, map_location=device))
    model_l.to(device)
    model_l.eval()
    
    process_images_with_players(
        args.input_folder, args.detection_file, model, model_l,
        args.kp_threshold, args.line_threshold, args.pnl_refine,
        args.save_path, args.display
    )
