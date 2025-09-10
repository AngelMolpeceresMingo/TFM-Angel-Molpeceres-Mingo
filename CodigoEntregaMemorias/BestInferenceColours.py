import cv2
import yaml
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torchvision.transforms.functional as f
import os

from tqdm import tqdm
from PIL import Image
from matplotlib.patches import Polygon

from model.cls_hrnet import get_cls_net
from model.cls_hrnet_l import get_cls_net as get_cls_net_l

from utils.utils_calib import FramebyFrameCalib, pan_tilt_roll_to_orientation
from utils.utils_heatmap import get_keypoints_from_heatmap_batch_maxpool, get_keypoints_from_heatmap_batch_maxpool_l, \
    complete_keypoints, coords_to_dict

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

def get_custom_line_color(line_index):
    
    custom_colors = {
        0: (0, 0, 255),       
        3: (0, 0, 255),      
        
        1: (0, 255, 255),   
        4: (0, 255, 255),      
        
        2: (255, 255, 0),    
        5: (255, 255, 0), 
        
        6: (139, 0, 0),   
        7: (139, 0, 0), 
        8: (139, 0, 0), 
        9: (139, 0, 0),  
        10: (139, 0, 0), 
        11: (139, 0, 0),
        
        12: (0, 255, 0),
        
        13: (0, 100, 0),  
        16: (0, 100, 0),   
        
        14: (0, 165, 255),    
        15: (0, 165, 255),  
        
        17: (221, 160, 221),  
        20: (221, 160, 221),  
        
        18: (203, 192, 255),  
        21: (203, 192, 255),  

        19: (42, 42, 165),     
        22: (42, 42, 165),     

        'circle_center': (255, 255, 0),     
        'arc_left': (255, 0, 255),         
        'arc_right': (255, 165, 0)         
    }
    
    return custom_colors.get(line_index, (128, 128, 128))

def get_custom_line_name(line_index):
    custom_names = {
        0: "Área 6m • Superior", 3: "Área 6m • Superior",  
        1: "Área 6m • Vertical", 4: "Área 6m • Vertical", 
        2: "Área 6m • Inferior", 5: "Área 6m • Inferior", 
        
        6: "Portería", 7: "Portería", 8: "Portería", 
        9: "Portería", 10: "Portería", 11: "Portería",
        
        12: "Centro Campo",                              
        13: "Banda Lateral", 16: "Banda Lateral",  
        14: "Línea de Gol", 15: "Línea de Gol",          
        
        17: "Área 18m • Superior", 20: "Área 18m • Superior",  
        18: "Área 18m • Vertical", 21: "Área 18m • Vertical",  
        19: "Área 18m • Inferior", 22: "Área 18m • Inferior"   
    }
    
    return custom_names.get(line_index, f"Línea {line_index}")

def is_line_visible(P, line, image_shape, min_length=10):
    height, width = image_shape[:2]
    
    w1 = line[0]
    w2 = line[1]
    
    i1 = P @ np.array([w1[0]-105/2, w1[1]-68/2, w1[2], 1])
    i2 = P @ np.array([w2[0]-105/2, w2[1]-68/2, w2[2], 1])
    
    if i1[-1] == 0 or i2[-1] == 0:
        return False
    
    i1 /= i1[-1]
    i2 /= i2[-1]
    
    margin = 0.1 * min(width, height)
    if ((-margin <= i1[0] <= width + margin and -margin <= i1[1] <= height + margin) or
        (-margin <= i2[0] <= width + margin and -margin <= i2[1] <= height + margin)):
        
        line_length = np.sqrt((i2[0]-i1[0])**2 + (i2[1]-i1[1])**2)
        if line_length > min_length:
            return True
    
    return False

def get_camera_orientation_by_posts(P, lines_coords, image_shape):

    left_goal_posts = [6, 7, 8]    
    right_goal_posts = [9, 10, 11]
    
    left_posts_visible = 0
    right_posts_visible = 0
    
    for i in left_goal_posts:
        if is_line_visible(P, lines_coords[i], image_shape):
            left_posts_visible += 1

    for i in right_goal_posts:
        if is_line_visible(P, lines_coords[i], image_shape):
            right_posts_visible += 1
    
    if left_posts_visible >= 2:
        return 'left_goal'
    elif right_posts_visible >= 2:
        return 'right_goal'
    else:
        return 'center'

def classify_line_by_slope_analysis(x1, y1, x2, y2, camera_orientation):
    if abs(x2 - x1) < 1e-6:
        return 'vertical'
    if abs(y2 - y1) < 1e-6: 
        return 'horizontal'
    
    if camera_orientation == 'left_goal':
        dx = x2 - x1
        dy = y2 - y1
        slope = dy / dx
        return 'vertical' if slope < 0 else 'horizontal'
        
    elif camera_orientation == 'right_goal':
        dx = x2 - x1
        dy = y2 - y1
        slope = dy / dx
        return 'vertical' if slope > 0 else 'horizontal'
        
    else:  
        angle_with_horizontal = np.arctan2(abs(y2 - y1), abs(x2 - x1)) * 180 / np.pi
        return 'horizontal' if angle_with_horizontal <= 7 else 'vertical'

def is_near_lines(projected_points, line_indices, P, lines_coords, proximity_threshold=50):

    if not projected_points:
        return False
    
    for line_idx in line_indices:
        if line_idx >= len(lines_coords):
            continue
            
        line = lines_coords[line_idx]
        w1, w2 = line[0], line[1]
        
        i1 = P @ np.array([w1[0]-105/2, w1[1]-68/2, w1[2], 1])
        i2 = P @ np.array([w2[0]-105/2, w2[1]-68/2, w2[2], 1])
        
        if i1[-1] == 0 or i2[-1] == 0:
            continue
            
        i1 /= i1[-1]
        i2 /= i2[-1]
        
        for point in projected_points[:10]:
            x, y = point
            
            A = i2[1] - i1[1]
            B = i1[0] - i2[0] 
            C = i2[0] * i1[1] - i1[0] * i2[1]
            
            if A*A + B*B > 0:
                distance = abs(A*x + B*y + C) / np.sqrt(A*A + B*B)
                if distance < proximity_threshold:
                    return True
    
    return False

def classify_circular_element(projected_points, P, lines_coords, camera_orientation):

    if not projected_points or len(projected_points) < 10:
        return 'unknown'
    
    center_line_idx = 12  
    left_area_lines = [0, 1, 2]  
    right_area_lines = [3, 4, 5]
    
    near_center = is_near_lines(projected_points, [center_line_idx], P, lines_coords, 80)
    
    near_left_area = is_near_lines(projected_points, left_area_lines, P, lines_coords, 60)
    near_right_area = is_near_lines(projected_points, right_area_lines, P, lines_coords, 60)
    
    points_array = np.array(projected_points)
    center_x = np.mean(points_array[:, 0])
    center_y = np.mean(points_array[:, 1])
    
    if near_center:
        return 'circle_center'
    elif near_left_area:
        return 'arc_left'
    elif near_right_area:
        return 'arc_right'
    else:
        if camera_orientation == 'left_goal':
            return 'arc_left' if center_x < 400 else 'circle_center'
        elif camera_orientation == 'right_goal':
            return 'arc_right' if center_x > 400 else 'circle_center'
        else:
            if center_x < 300:
                return 'arc_left'
            elif center_x > 500:
                return 'arc_right'
            else:
                return 'circle_center'

def draw_custom_color_legend_updated(frame_output, height, width):
    legend_groups = [
        ("Áreas 6m", [(0, "Superior • Rojo"), (1, "Vertical • Amarillo"), (2, "Inferior • Azul F.")]),
        ("Porterías", [(6, "Postes • Azul Osc.")]),
        ("Principales", [(12, "Centro • Verde F."), (13, "Bandas • Verde Osc."), (14, "Goles • Naranja")]),
        ("Áreas 18m", [(17, "Superior • Morado"), (18, "Vertical • Rosa"), (19, "Inferior • Marrón")]),
        ("Circulares", [('circle_center', "Centro • Cian"), ('arc_left', "Arco Izq • Magenta"), ('arc_right', "Arco Der • Azul Claro")])
    ]
    
    legend_x = width - 220 
    legend_y = 25
    current_y = legend_y + 5
    
    total_height = sum(30 + len(lines) * 18 for _, lines in legend_groups) + 30
    
    overlay = frame_output.copy()
    
    for i in range(8):
        margin = i * 2
        cv2.rectangle(overlay, 
                     (legend_x - 15 + margin, legend_y - 5 + margin), 
                     (width - 8 - margin, legend_y + total_height - margin), 
                     (20 + i*5, 20 + i*5, 40 + i*10), -1)
    
    cv2.addWeighted(overlay, 0.75, frame_output, 0.25, 0, frame_output)
    
    cv2.putText(frame_output, "COLORES PERSONALIZADOS+", (legend_x - 5, current_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
    cv2.putText(frame_output, "COLORES PERSONALIZADOS+", (legend_x - 5, current_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
    current_y += 25
    
    for group_name, lines_list in legend_groups:
        cv2.putText(frame_output, group_name, (legend_x - 8, current_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 255), 1)
        current_y += 22
        
        for line_idx, line_name in lines_list:
            color = get_custom_line_color(line_idx)
            
            cv2.circle(frame_output, (legend_x + 8, current_y - 3), 4, color, -1)
            cv2.circle(frame_output, (legend_x + 8, current_y - 3), 4, (255, 255, 255), 1)
            
            cv2.putText(frame_output, line_name, (legend_x + 20, current_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.32, (240, 240, 240), 1)
            current_y += 16
        
        current_y += 8  

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
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)

    frame = f.to_tensor(frame).float().unsqueeze(0)
    _, _, h_original, w_original = frame.size()
    frame = frame if frame.size()[-1] == 960 else transform2(frame)
    frame = frame.to(device)
    b, c, h, w = frame.size()

    with torch.no_grad():
        heatmaps = model(frame)
        heatmaps_l = model_l(frame)

    kp_coords = get_keypoints_from_heatmap_batch_maxpool(heatmaps[:,:-1,:,:])
    line_coords = get_keypoints_from_heatmap_batch_maxpool_l(heatmaps_l[:,:-1,:,:])
    kp_dict = coords_to_dict(kp_coords, threshold=kp_threshold)
    lines_dict = coords_to_dict(line_coords, threshold=line_threshold)
    kp_dict, lines_dict = complete_keypoints(kp_dict[0], lines_dict[0], w=w, h=h, normalize=True)

    cam.update(kp_dict, lines_dict)
    final_params_dict = cam.heuristic_voting(refine_lines=pnl_refine)

    return final_params_dict

def project(frame, P):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_output = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
    
    height, width = frame.shape[:2]
    
    camera_orientation = get_camera_orientation_by_posts(P, lines_coords, frame.shape)
    
    left_posts = sum(1 for i in [6, 7, 8] if is_line_visible(P, lines_coords[i], frame.shape))
    right_posts = sum(1 for i in [9, 10, 11] if is_line_visible(P, lines_coords[i], frame.shape))
    
    orientation_text = {
        'left_goal': f'Portería Alpha',
        'right_goal': f'Portería Beta',
        'center': f'Vista Central'
    }
    
    debug_text = f"{orientation_text[camera_orientation]}"
    debug_text2 = f"Colores Personalizados • α={left_posts} • β={right_posts}"
    
    cv2.putText(frame_output, debug_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
    cv2.putText(frame_output, debug_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    cv2.putText(frame_output, debug_text2, (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
    cv2.putText(frame_output, debug_text2, (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    
    detected_lines = []
    
    for i, line in enumerate(lines_coords):
        w1 = line[0]
        w2 = line[1]
        i1 = P @ np.array([w1[0]-105/2, w1[1]-68/2, w1[2], 1])
        i2 = P @ np.array([w2[0]-105/2, w2[1]-68/2, w2[2], 1])
        
        if i1[-1] != 0 and i2[-1] != 0:
            i1 /= i1[-1]
            i2 /= i2[-1]
            
            color = get_custom_line_color(i)
            line_name = get_custom_line_name(i)
            
            orientation = classify_line_by_slope_analysis(
                i1[0], i1[1], i2[0], i2[1], camera_orientation
            )
            
            line_drawn = False
            
            if i in [13, 16]:
                margin = 10
                if (-width*margin <= i1[0] <= width*margin and -height*margin <= i1[1] <= height*margin and
                    -width*margin <= i2[0] <= width*margin and -height*margin <= i2[1] <= height*margin):
                    cv2.line(frame_output, (int(i1[0])+1, int(i1[1])+1), (int(i2[0])+1, int(i2[1])+1), (0, 0, 0), 6)
                    cv2.line(frame_output, (int(i1[0]), int(i1[1])), (int(i2[0]), int(i2[1])), color, 5)
                    line_drawn = True
                              
            elif i in [14, 15]:
                margin = 3
                if (-width*margin <= i1[0] <= width*margin and -height*margin <= i1[1] <= height*margin and
                    -width*margin <= i2[0] <= width*margin and -height*margin <= i2[1] <= height*margin):
                    cv2.line(frame_output, (int(i1[0])+1, int(i1[1])+1), (int(i2[0])+1, int(i2[1])+1), (0, 0, 0), 5)
                    cv2.line(frame_output, (int(i1[0]), int(i1[1])), (int(i2[0]), int(i2[1])), color, 4)
                    line_drawn = True
            else:
                
                if (-width <= i1[0] <= width*2 and -height <= i1[1] <= height*2 and
                    -width <= i2[0] <= width*2 and -height <= i2[1] <= height*2):
                    cv2.line(frame_output, (int(i1[0])+1, int(i1[1])+1), (int(i2[0])+1, int(i2[1])+1), (0, 0, 0), 4)
                    cv2.line(frame_output, (int(i1[0]), int(i1[1])), (int(i2[0]), int(i2[1])), color, 3)
                    line_drawn = True
            
            if line_drawn:
                detected_lines.append((i, line_name, orientation))

    r = 9.15
    circular_elements = []
    
    pts_left = []
    base_pos = np.array([11-105/2, 68/2-68/2, 0., 0.])
    for ang in np.linspace(37, 143, 50):
        ang = np.deg2rad(ang)
        pos = base_pos + np.array([r*np.sin(ang), r*np.cos(ang), 0., 1.])
        ipos = P @ pos
        if ipos[-1] != 0:
            ipos /= ipos[-1]
            pts_left.append([ipos[0], ipos[1]])

    pts_right = []
    base_pos = np.array([94-105/2, 68/2-68/2, 0., 0.])
    for ang in np.linspace(217, 323, 50):
        ang = np.deg2rad(ang)
        pos = base_pos + np.array([r*np.sin(ang), r*np.cos(ang), 0., 1.])
        ipos = P @ pos
        if ipos[-1] != 0:
            ipos /= ipos[-1]
            pts_right.append([ipos[0], ipos[1]])

    pts_center = []
    base_pos = np.array([0, 0, 0., 0.])
    for ang in np.linspace(0, 360, 100):
        ang = np.deg2rad(ang)
        pos = base_pos + np.array([r*np.sin(ang), r*np.cos(ang), 0., 1.])
        ipos = P @ pos
        if ipos[-1] != 0:
            ipos /= ipos[-1]
            pts_center.append([ipos[0], ipos[1]])

    circular_elements_data = [
        (pts_left, "Arco Izquierdo"),
        (pts_right, "Arco Derecho"), 
        (pts_center, "Círculo Central")
    ]
    
    detected_circles = []
    
    for pts, default_name in circular_elements_data:
        if pts:
            element_type = classify_circular_element(pts, P, lines_coords, camera_orientation)
            
            color = get_custom_line_color(element_type)
            
            element_names = {
                'circle_center': 'Círculo Central',
                'arc_left': 'Arco Izquierdo', 
                'arc_right': 'Arco Derecho',
                'unknown': f'{default_name}'
            }
            element_name = element_names.get(element_type, default_name)
            
            pts_array = np.array(pts, np.int32)
            shadow_pts = pts_array + np.array([2, 2])
            
            thickness = 4 if element_type == 'circle_center' else 3
            shadow_thickness = thickness + 1
            
            frame_output = cv2.polylines(frame_output, [shadow_pts], False, (0, 0, 0), shadow_thickness)
            frame_output = cv2.polylines(frame_output, [pts_array], False, color, thickness)

            detected_circles.append((element_type, element_name))

    stats_y = height - 150 
    stats_bg = frame_output.copy()
    cv2.rectangle(stats_bg, (5, stats_y - 15), (400, height - 5), (0, 0, 0), -1)
    cv2.addWeighted(stats_bg, 0.7, frame_output, 0.3, 0, frame_output)

    cv2.putText(frame_output, f"Líneas: {len(detected_lines)} | Círculos: {len(detected_circles)}", 
               (15, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    circle_start_y = stats_y + 25 + min(len(detected_lines), 2) * 25
    for idx, (element_type, element_name) in enumerate(detected_circles[:3]):
        if circle_start_y + idx * 25 < height - 15:
            color = get_custom_line_color(element_type)
            cv2.putText(frame_output, element_name, (15, circle_start_y + idx * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

    draw_custom_color_legend_updated(frame_output, height, width)

    return frame_output

def process_input(input_path, input_type, model_kp, model_line, kp_threshold, line_threshold, pnl_refine,
                  save_path, display):

    cap = cv2.VideoCapture(input_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    cam = FramebyFrameCalib(iwidth=frame_width, iheight=frame_height, denormalize=True)

    if input_type == 'image':
        frame = cv2.imread(input_path)
        if frame is None:
            print(f"Error: Unable to read the image {input_path}")
            return

        final_params_dict = inference(cam, frame, model, model_l, kp_threshold, line_threshold, pnl_refine)
        if final_params_dict is not None:
            P = projection_from_cam_params(final_params_dict)
            projected_frame = project(frame, P)
        else:
            projected_frame = frame

        if save_path != "":
            cv2.imwrite(save_path, projected_frame)
        else:
            plt.imshow(cv2.cvtColor(projected_frame, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process video or image and plot lines on each frame.")
    parser.add_argument("--weights_kp", type=str, help="Path to the model for keypoint inference.")
    parser.add_argument("--weights_line", type=str, help="Path to the model for line projection.")
    parser.add_argument("--kp_threshold", type=float, default=0.3434, help="Threshold for keypoint detection.")
    parser.add_argument("--line_threshold", type=float, default=0.7867, help="Threshold for line detection.")
    parser.add_argument("--pnl_refine", action="store_true", help="Enable PnL refinement module.")
    parser.add_argument("--device", type=str, default="cpu", help="CPU or CUDA device index")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input video or image file.")
    parser.add_argument("--input_type", type=str, choices=['video', 'image'], required=True,
                        help="Type of input: 'video' or 'image'.")
    parser.add_argument("--save_path", type=str, default="", help="Path to save the processed video.")
    parser.add_argument("--display", action="store_true", help="Enable real-time display.")
    args = parser.parse_args()

    input_path = args.input_path
    input_type = args.input_type
    model_kp = args.weights_kp
    model_line = args.weights_line
    pnl_refine = args.pnl_refine
    save_path = args.save_path
    device = args.device
    display = args.display and input_type == 'video'
    kp_threshold = args.kp_threshold
    line_threshold = args.line_threshold

    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    config_path_kp = os.path.join(script_dir, "config", "hrnetv2_w48.yaml")
    config_path_line = os.path.join(script_dir, "config", "hrnetv2_w48_l.yaml")
    
    weights_path_kp = os.path.join(script_dir, "weights", args.weights_kp)
    weights_path_line = os.path.join(script_dir, "weights", args.weights_line)

    cfg = yaml.safe_load(open(config_path_kp, 'r'))
    cfg_l = yaml.safe_load(open(config_path_line, 'r'))

    loaded_state = torch.load(weights_path_kp, map_location=device)
    model = get_cls_net(cfg)
    model.load_state_dict(loaded_state)
    model.to(device)
    model.eval()

    loaded_state_l = torch.load(weights_path_line, map_location=device)
    model_l = get_cls_net_l(cfg_l)
    model_l.load_state_dict(loaded_state_l)
    model_l.to(device)
    model_l.eval()

    transform2 = T.Resize((540, 960))

    process_input(input_path, input_type, model_kp, model_line, kp_threshold, line_threshold, pnl_refine,
                  save_path, display)
