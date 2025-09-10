import cv2
import yaml
import torch
import argparse
import numpy as np
import os

from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as f

from model.cls_hrnet import get_cls_net
from model.cls_hrnet_l import get_cls_net as get_cls_net_l

from utils.utils_calib import FramebyFrameCalib
from utils.utils_heatmap import get_keypoints_from_heatmap_batch_maxpool, get_keypoints_from_heatmap_batch_maxpool_l, \
    complete_keypoints, coords_to_dict

lines_coords = [
    [[0., 54.16, 0.], [16.5, 54.16, 0.]], 
    [[16.5, 13.84, 0.], [16.5, 54.16, 0.]], 
    [[16.5, 13.84, 0.], [0., 13.84, 0.]],
    [[88.5, 54.16, 0.], [105., 54.16, 0.]], 
    [[88.5, 13.84, 0.], [88.5, 54.16, 0.]], 
    [[88.5, 13.84, 0.], [105., 13.84, 0.]],
    [[0., 37.66, -2.44], [0., 30.34, -2.44]], [[0., 37.66, 0.], [0., 37.66, -2.44]], [[0., 30.34, 0.], [0., 30.34, -2.44]],
    [[105., 37.66, -2.44], [105., 30.34, -2.44]], [[105., 30.34, 0.], [105., 30.34, -2.44]], [[105., 37.66, 0.], [105., 37.66, -2.44]],

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
    [[99.5, 24.84, 0.], [105., 24.84, 0.]]
]

def get_custom_line_color(line_index):
    custom_colors = {
        0: (221, 160, 221), 3: (221, 160, 221), 2: (42, 42, 165), 5: (42, 42, 165),
        1: (203, 192, 255), 4: (203, 192, 255),
        17: (0, 0, 255), 20: (0, 0, 255), 19: (255, 255, 0), 22: (255, 255, 0),
        18: (0, 255, 255), 21: (0, 255, 255),
        12: (0, 255, 0), 13: (0, 100, 0), 16: (0, 100, 0),
        14: (0, 165, 255), 15: (0, 165, 255),
        'circle_center': (255, 255, 0), 'arc_left': (255, 0, 255), 'arc_right': (255, 165, 0),
        'virtual': (192, 192, 192)
    }
    return custom_colors.get(line_index, (128, 128, 128))

def get_keypoint_color_by_line_relation(related_lines):
    if not related_lines: return (255, 255, 255)
    if 'virtual' in related_lines: return get_custom_line_color('virtual')
    priority_order = [[17,18,19,20,21,22], [0,1,2,3,4,5], [12,13,14,15,16]]
    for group in priority_order:
        for line_idx in related_lines:
            if line_idx in group: return get_custom_line_color(line_idx)
    if 'arc_left' in related_lines: return get_custom_line_color('arc_left')
    if 'arc_right' in related_lines: return get_custom_line_color('arc_right')
    if 'circle_center' in related_lines: return get_custom_line_color('circle_center')
    return get_custom_line_color(related_lines[0])

def is_line_visible(P, line, image_shape, min_length=10):
    height, width = image_shape[:2]
    w1, w2 = line[0], line[1]
    i1 = P @ np.array([w1[0]-52.5, w1[1]-34, w1[2], 1]); i2 = P @ np.array([w2[0]-52.5, w2[1]-34, w2[2], 1])
    if i1[-1] == 0 or i2[-1] == 0: return False
    i1 /= i1[-1]; i2 /= i2[-1]
    margin = 0.3 * min(width, height)
    if ((-margin <= i1[0] <= width + margin and -margin <= i1[1] <= height + margin) or
        (-margin <= i2[0] <= width + margin and -margin <= i2[1] <= height + margin)):
        if np.sqrt((i2[0]-i1[0])**2 + (i2[1]-i1[1])**2) > min_length: return True
    return False

def get_projected_line_points(P, line):
    w1, w2 = line[0], line[1]
    i1 = P @ np.array([w1[0]-52.5, w1[1]-34, w1[2], 1]); i2 = P @ np.array([w2[0]-52.5, w2[1]-34, w2[2], 1])
    if i1[-1] == 0 or i2[-1] == 0: return None
    i1 /= i1[-1]; i2 /= i2[-1]
    return [(i1[0], i1[1]), (i2[0], i2[1])]

def calculate_line_intersection(line1_points, line2_points):
    if not line1_points or not line2_points: return None
    (x1, y1), (x2, y2) = line1_points
    (x3, y3), (x4, y4) = line2_points
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10: return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

def calculate_line_midpoint(p1, p2):
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

def calculate_arc_cusp_point(P, arc_center_3d, radius, cusp_direction):
    cusp_3d = [arc_center_3d[0] + radius * cusp_direction[0], arc_center_3d[1] + radius * cusp_direction[1], arc_center_3d[2]]
    cusp_homo = P @ np.array([cusp_3d[0]-52.5, cusp_3d[1]-34, cusp_3d[2], 1])
    if cusp_homo[2] == 0: return None
    return (cusp_homo[0] / cusp_homo[2], cusp_homo[1] / cusp_homo[2])

def add_circle_points(P, keypoints):
    center, radius, type, name = [52.5, 34, 0], 9.15, "circle_center", "Circulo"
    for i, direction in enumerate([[1, 0], [-1, 0], [0, 1], [0, -1]]):
        endpoint = calculate_arc_cusp_point(P, center, radius, direction)
        if endpoint:
            dir_name = ["Der", "Izq", "Sup", "Inf"][i]
            keypoints.append({'name': f'{type}_endpoint_{i}', 'point': endpoint, 'related_lines': [type], 'description': f'{name} {dir_name}'})

def add_midfield_band_endpoints(P, lines_coords, projected_lines, keypoints, image_shape):
    height, width = image_shape[:2]
    margin = 0.2 * min(width, height)
    if 12 in projected_lines:
        line_points = projected_lines[12]
        (x1, y1), (x2, y2) = line_points
        for i, (x, y) in enumerate([(x1, y1), (x2, y2)]):
            if (-margin <= x <= width + margin and -margin <= y <= height + margin):
                extreme = "Banda-Sup" if i == 0 else "Banda-Inf"
                keypoints.append({
                    'name': f'midfield_endpoint_{i}',
                    'point': (x, y),
                    'related_lines': [12],
                    'description': f'Centro {extreme}'
                })

def project_single_point(P, point_3d):
    point_homo = P @ np.array([point_3d[0] - 52.5, point_3d[1] - 34, point_3d[2], 1])
    if point_homo[2] == 0: return None
    return (point_homo[0] / point_homo[2], point_homo[1] / point_homo[2])

def add_extended_sideline_points(P, keypoints, image_shape):
    height, width = image_shape[:2]
    margin = 0.1 * min(width, height)
    virtual_points_3d = {
        "Ext Area-I Banda-Inf": (16.5, 0, 0), "Ext Area-I Banda-Sup": (16.5, 68, 0),
        "Ext Area-D Banda-Inf": (88.5, 0, 0), "Ext Area-D Banda-Sup": (88.5, 68, 0),
        "Ext Circ-I Banda-Inf": (52.5 - 9.15, 0, 0), "Ext Circ-I Banda-Sup": (52.5 - 9.15, 68, 0),
        "Ext Circ-D Banda-Inf": (52.5 + 9.15, 0, 0), "Ext Circ-D Banda-Sup": (52.5 + 9.15, 68, 0),
    }
    for name, point_3d in virtual_points_3d.items():
        point_2d = project_single_point(P, point_3d)
        if point_2d and -margin <= point_2d[0] <= width + margin and -margin <= point_2d[1] <= height + margin:
            keypoints.append({'name': name.lower().replace(" ", "_"), 'point': point_2d, 'related_lines': ['virtual'], 'description': name})

def extract_keypoints_from_lines(P, lines_coords, image_shape):
    keypoints = []
    projected_lines = {i: get_projected_line_points(P, line) for i, line in enumerate(lines_coords) if is_line_visible(P, line, image_shape)}

    intersection_defs = [
        (17, 18, '6m Izq-Sup Inter'), (18, 19, '6m Izq-Inf Inter'), (20, 21, '6m Der-Sup Inter'), (21, 22, '6m Der-Inf Inter'),
        (0, 1, '18m Izq-Sup Inter'), (1, 2, '18m Izq-Inf Inter'), (3, 4, '18m Der-Sup Inter'), (4, 5, '18m Der-Inf Inter'),
        (12, 13, 'Centro-Banda Sup'), (12, 16, 'Centro-Banda Inf'),
        (13, 14, 'Esquina Izq-Sup'), (13, 15, 'Esquina Der-Sup'), (16, 14, 'Esquina Izq-Inf'), (16, 15, 'Esquina Der-Inf')
    ]
    for l1, l2, desc in intersection_defs:
        if l1 in projected_lines and l2 in projected_lines:
            intersection = calculate_line_intersection(projected_lines[l1], projected_lines[l2])
            if intersection: keypoints.append({'name': desc.lower().replace(" ", "_"), 'point': intersection, 'related_lines': [l1, l2], 'description': desc})
    
    area_goalline_pairs = [
        (17, 14, "6m Sup-Fondo Izq"), (19, 14, "6m Inf-Fondo Izq"), (0, 14, "18m Sup-Fondo Izq"), (2, 14, "18m Inf-Fondo Izq"),
        (20, 15, "6m Sup-Fondo Der"), (22, 15, "6m Inf-Fondo Der"), (3, 15, "18m Sup-Fondo Der"), (5, 15, "18m Inf-Fondo Der")
    ]
    for l1, l2, desc in area_goalline_pairs:
        if l1 in projected_lines and l2 in projected_lines:
            intersection = calculate_line_intersection(projected_lines[l1], projected_lines[l2])
            if intersection: keypoints.append({'name': desc.lower().replace(" ","_"), 'point': intersection, 'related_lines': [l1, l2], 'description': desc})

    midpoint_defs = {
        0: "18m Sup-H Medio", 2: "18m Inf-H Medio", 3: "18m Sup-H Medio", 5: "18m Inf-H Medio",
        17: "6m Sup-H Medio", 19: "6m Inf-H Medio", 20: "6m Sup-H Medio", 22: "6m Inf-H Medio",
        18: "6m V Medio Izq", 21: "6m V Medio Der", 13: "Banda Sup Medio", 16: "Banda Inf Medio"
    }
    for line_idx, desc in midpoint_defs.items():
        if line_idx in projected_lines:
            p1, p2 = projected_lines[line_idx]
            keypoints.append({'name': f'midpoint_{line_idx}', 'point': calculate_line_midpoint(p1, p2), 'related_lines': [line_idx], 'description': desc})
    
    if 12 in projected_lines:
        p1, p2 = projected_lines[12]
        keypoints.append({'name': 'field_center', 'point': calculate_line_midpoint(p1, p2), 'related_lines': [12], 'description': 'Centro Campo'})

    arc_cusps = [([11, 34, 0], 9.15, [1, 0], "Arco Izq Cúspide"), ([94, 34, 0], 9.15, [-1, 0], "Arco Der Cúspide")]
    for center, radius, direction, desc in arc_cusps:
        cusp = calculate_arc_cusp_point(P, center, radius, direction)
        if cusp: keypoints.append({'name': desc.lower().replace(" ", "_"), 'point': cusp, 'related_lines': ['arc_left' if 'Izq' in desc else 'arc_right'], 'description': desc})

    add_circle_points(P, keypoints)
    add_midfield_band_endpoints(P, lines_coords, projected_lines, keypoints, image_shape)

    add_extended_sideline_points(P, keypoints, image_shape)

    height, width = image_shape[:2]
    margin = 0.2 * min(width, height)
    return [kp for kp in keypoints if kp['point'] and -margin <= kp['point'][0] <= width + margin and -margin <= kp['point'][1] <= height + margin]

def draw_keypoints_legend(frame_output, height, width):
    legend_groups = [
        ("Área Pequeña", [("Intersección", (0, 0, 255)), ("Inter-Fondo", (0, 255, 255)), ("Medio Vertical", (0, 255, 255))]),
        ("Área Grande", [("Intersección", (221, 160, 221)), ("Medio Horizontal", (42, 42, 165))]),
        ("Campo", [("Centro/Esquina", (0, 255, 0)), ("Medio Banda", (0, 100, 0)), ("Inter Centro-Banda", (0, 255, 0)), ("Extremos Centro", (0, 255, 0))]),
        ("Arcos", [("Cúspide Arco", (255, 0, 255)), ("Puntos Círculo", (255, 255, 0))]),
        ("Virtuales", [("Extensión a Banda", get_custom_line_color('virtual'))])
    ]
    legend_x, current_y = width - 260, 30
    total_height = sum(25 + len(points) * 16 for _, points in legend_groups) + 20
    overlay = frame_output.copy()
    cv2.rectangle(overlay, (legend_x - 10, 20), (width - 5, 25 + total_height), (20, 20, 40, 180), -1)
    cv2.addWeighted(overlay, 0.7, frame_output, 0.3, 0, frame_output)
    cv2.putText(frame_output, "PUNTOS CLAVE (COMPLETO)", (legend_x - 5, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1); current_y += 20
    for group, points in legend_groups:
        cv2.putText(frame_output, group, (legend_x - 5, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 255), 1); current_y += 18
        for p_type, color in points:
            cv2.circle(frame_output, (legend_x + 8, current_y - 3), 3, color, -1)
            cv2.circle(frame_output, (legend_x + 8, current_y - 3), 3, (255, 255, 255), 1)
            cv2.putText(frame_output, p_type, (legend_x + 18, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (240, 240, 240), 1); current_y += 14
        current_y += 5


def projection_from_cam_params(final_params_dict):
    cam_params = final_params_dict["cam_params"]
    x_f, y_f = cam_params['x_focal_length'], cam_params['y_focal_length']
    p_p = np.array(cam_params['principal_point'])
    pos_m = np.array(cam_params['position_meters'])
    rot = np.array(cam_params['rotation_matrix'])
    It = np.eye(4)[:-1]; It[:, -1] = -pos_m
    Q = np.array([[x_f, 0, p_p[0]], [0, y_f, p_p[1]], [0, 0, 1]])
    return Q @ (rot @ It)

def inference(cam, frame, model, model_l, kp_threshold, line_threshold, pnl_refine, device, transform2):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    frame_tensor = f.to_tensor(frame_pil).float().unsqueeze(0)
    if frame_tensor.size(-1) != 960: frame_tensor = transform2(frame_tensor)
    frame_tensor = frame_tensor.to(device)
    with torch.no_grad():
        heatmaps = model(frame_tensor)
        heatmaps_l = model_l(frame_tensor)
    w, h = frame_tensor.size(-1), frame_tensor.size(-2)
    kp_coords = get_keypoints_from_heatmap_batch_maxpool(heatmaps[:,:-1,:,:])
    line_coords = get_keypoints_from_heatmap_batch_maxpool_l(heatmaps_l[:,:-1,:,:])
    kp_dict = coords_to_dict(kp_coords, threshold=kp_threshold)
    lines_dict = coords_to_dict(line_coords, threshold=line_threshold)
    kp_dict, lines_dict = complete_keypoints(kp_dict[0], lines_dict[0], w=w, h=h, normalize=True)
    cam.update(kp_dict, lines_dict)
    return cam.heuristic_voting(refine_lines=pnl_refine)

def project_keypoints_from_lines(frame, P):
    frame_output = frame.copy()
    height, width = frame.shape[:2]
    keypoints = extract_keypoints_from_lines(P, lines_coords, frame.shape)
    
    for kp in keypoints:
        kp['color'] = get_keypoint_color_by_line_relation(kp['related_lines'])
        x, y = int(kp['point'][0]), int(kp['point'][1])
        name = kp['description']
        
        cv2.circle(frame_output, (x, y), 8, kp['color'], -1)
        cv2.circle(frame_output, (x, y), 8, (255, 255, 255), 2)
        cv2.circle(frame_output, (x, y), 3, (0, 0, 0), -1)
        
        text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
        label_x, label_y = x + 12, y + 5
        cv2.rectangle(frame_output, (label_x - 2, label_y - text_size[1] - 3), (label_x + text_size[0] + 2, label_y + 3), (0, 0, 0, 180), -1)
        cv2.putText(frame_output, name, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    draw_keypoints_legend(frame_output, height, width)
    return frame_output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calcula y dibuja puntos clave desde líneas del campo.")
    parser.add_argument("--weights_kp", type=str, required=True)
    parser.add_argument("--weights_line", type=str, required=True)
    parser.add_argument("--kp_threshold", type=float, default=0.3)
    parser.add_argument("--line_threshold", type=float, default=0.3)
    parser.add_argument("--pnl_refine", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    config_path_kp = os.path.join(script_dir, "config", "hrnetv2_w48.yaml")
    config_path_line = os.path.join(script_dir, "config", "hrnetv2_w48_l.yaml")
    
    cfg = yaml.safe_load(open(config_path_kp, 'r')); model = get_cls_net(cfg)
    model.load_state_dict(torch.load(os.path.join(script_dir, "weights", args.weights_kp), map_location=device)); model.to(device).eval()
    
    cfg_l = yaml.safe_load(open(config_path_line, 'r')); model_l = get_cls_net_l(cfg_l)
    model_l.load_state_dict(torch.load(os.path.join(script_dir, "weights", args.weights_line), map_location=device)); model_l.to(device).eval()

    transform2 = T.Resize((540, 960))
    frame = cv2.imread(args.input_path)
    if frame is None: raise FileNotFoundError(f"No se pudo leer la imagen en: {args.input_path}")

    cam = FramebyFrameCalib(iwidth=frame.shape[1], iheight=frame.shape[0], denormalize=True)
    final_params_dict = inference(cam, frame, model, model_l, args.kp_threshold, args.line_threshold, args.pnl_refine, device, transform2)
    
    if final_params_dict:
        P = projection_from_cam_params(final_params_dict)
        projected_frame = project_keypoints_from_lines(frame, P)
    else:
        projected_frame = frame
        print("Advertencia: No se pudo calibrar la cámara. Mostrando imagen original.")

    cv2.imwrite(args.save_path, projected_frame)
    print(f"Imagen procesada y guardada en: {args.save_path}")

