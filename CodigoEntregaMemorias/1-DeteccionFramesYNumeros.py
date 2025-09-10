import os
import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm
from types import SimpleNamespace
import torch
from sklearn.cluster import KMeans

try:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
except pytesseract.TesseractNotFoundError:
    print("ADVERTENCIA: Tesseract no se encontró en la ruta por defecto. "
          "Asegúrate de que esté instalado y de que la ruta en el script sea correcta.")

COLORS_BASIC = {
    'blanco': [255, 255, 255], 'negro': [0, 0, 0], 'rojo': [255, 0, 0],
    'verde_cesped': [34, 139, 34], 'verde': [0, 128, 0], 'verde_claro': [50, 205, 50],
    'azul': [0, 0, 255], 'amarillo': [255, 255, 0], 'naranja': [255, 165, 0],
    'morado': [128, 0, 128], 'marrón': [139, 69, 19], 'gris': [128, 128, 128]
}
SIMILARITY_THRESHOLD = 200

def crop_torso(player_image):
    altura, ancho = player_image.shape[:2]
    inicio_y = int(altura * 0.2)
    fin_y = int(altura * 0.5)
    return player_image[inicio_y:fin_y, 0:ancho]

def improve_resolution(image):
    if image.size == 0: return image
    height, width = image.shape[:2]
    new_size = (width * 2, height * 2)
    return cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)

def enhance_colors_selectively(image, saturation_boost=2.0, contrast_alpha=1.3, gray_threshold=30):
    if image.size == 0: return image
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_enhanced = cv2.convertScaleAbs(l, alpha=contrast_alpha, beta=0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_enhanced)
    a_int = a.astype(np.int16) - 128
    b_int = b.astype(np.int16) - 128
    color_intensity = np.sqrt(a_int**2 + b_int**2).astype(np.uint8)
    _, color_mask = cv2.threshold(color_intensity, gray_threshold, 255, cv2.THRESH_BINARY)
    mask_float = (color_mask / 255.0).astype(np.float32)
    s_float = s.astype(np.float32)
    s_enhanced = s_float * (1.0 + saturation_boost * mask_float)
    s_enhanced = np.clip(s_enhanced, 0, 255).astype(np.uint8)
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    hsv_enhanced = cv2.merge([h, s_enhanced, v])
    result_lab = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    result_hsv = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
    return cv2.addWeighted(result_hsv, 0.6, result_lab, 0.4, 0)

def is_green_dominant(pixel):
    return pixel[1] > pixel[0] and pixel[1] > pixel[2] and pixel[1] > 60

def find_closest_basic_color_step1(pixel, basic_colors, threshold):
    pixel = np.array(pixel, dtype=np.float32)
    if is_green_dominant(pixel):
        return np.array(basic_colors['verde_cesped'], dtype=np.float32)
    min_dist = float('inf')
    closest_color = None
    for color_name, color_rgb in basic_colors.items():
        if 'verde' in color_name: continue
        color_rgb = np.array(color_rgb, dtype=np.float32)
        dist = np.sqrt(np.sum((pixel - color_rgb) ** 2))
        if dist < min_dist and dist < threshold:
            min_dist = dist
            closest_color = color_rgb
    return closest_color if closest_color is not None else pixel

def simplify_colors_step1(image):
    if image.size == 0: return image
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, channels = image_rgb.shape
    pixels = image_rgb.reshape(-1, channels)
    kmeans = KMeans(n_clusters=8, random_state=0, n_init=10).fit(pixels)
    centers = kmeans.cluster_centers_
    result = np.zeros_like(image_rgb)
    for i in range(height):
        for j in range(width):
            pixel = kmeans.cluster_centers_[kmeans.labels_[i*width + j]]
            homogenized_color = find_closest_basic_color_step1(pixel, COLORS_BASIC, SIMILARITY_THRESHOLD)
            result[i, j] = homogenized_color
    return cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_RGB2BGR)

def find_closest_basic_color_step2(pixel, basic_colors):
    pixel = np.array(pixel, dtype=np.float32)
    min_dist = float('inf')
    closest_color = None
    for color_rgb in basic_colors.values():
        color_rgb = np.array(color_rgb, dtype=np.float32)
        dist = np.linalg.norm(pixel - color_rgb)
        if dist < min_dist:
            min_dist = dist
            closest_color = color_rgb
    return closest_color

def simplify_colors_step2(image):
    if image.size == 0: return image
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, channels = image_rgb.shape
    pixels = image_rgb.reshape(-1, channels)
    result_pixels = np.array([find_closest_basic_color_step2(p, COLORS_BASIC) for p in pixels])
    result_image_rgb = result_pixels.reshape(height, width, channels).astype(np.uint8)
    return cv2.cvtColor(result_image_rgb, cv2.COLOR_RGB2BGR)

def isolate_color_components(image):
    if image.size == 0: return []
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image_rgb.shape
    center = (width // 2, height // 2)
    pixels = image_rgb.reshape(-1, 3)
    coords = [(i % width, i // width) for i in range(len(pixels))]
    distances = [np.sqrt((x - center[0])**2 + (y - center[1])**2) for x, y in coords]
    color_dist = sorted(list(zip([tuple(p) for p in pixels], distances)), key=lambda x: x[1])
    seen = set()
    ordered_colors = [c for c, d in color_dist if c not in seen and not seen.add(c)]
    
    binary_images = []
    for color in ordered_colors[:4]:
        binary_img = np.ones((height, width), dtype=np.uint8) * 255
        mask = np.all(image_rgb == color, axis=2)
        binary_img[mask] = 0
        binary_images.append(binary_img)
    
    while len(binary_images) < 4:
        binary_images.append(np.ones((height, width), dtype=np.uint8) * 255)
    return binary_images

def filter_number_candidates(binary_img):
    if binary_img.size == 0 or np.all(binary_img == 255): return binary_img
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cv2.bitwise_not(binary_img), 4, cv2.CV_32S)
    
    if num_labels <= 1: return np.ones_like(binary_img) * 255

    areas = stats[1:, cv2.CC_STAT_AREA]
    valid_indices = np.where(areas > 0)[0] + 1
    
    if len(valid_indices) == 0: return np.ones_like(binary_img) * 255
        
    height, width = binary_img.shape
    non_border_indices = []
    for i in valid_indices:
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if x > 0 and y > 0 and (x + w) < width and (y + h) < height:
            non_border_indices.append(i)

    if not non_border_indices: return np.ones_like(binary_img) * 255
    
    non_border_areas = stats[non_border_indices, cv2.CC_STAT_AREA]
    largest_indices = np.array(non_border_indices)[np.argsort(non_border_areas)[-2:]]

    filtered_img = np.ones_like(binary_img) * 255
    for i in largest_indices:
        filtered_img[labels == i] = 0
        
    return filtered_img

def perform_ocr_on_component(image):
    if image is None or image.size == 0 or np.all(image == 255):
        return None
    
    h, w = image.shape
    if h == 0 or w == 0: return None
    escala = 300 / h
    nuevo_w, nuevo_h = int(w * escala), int(h * escala)
    img_redim = cv2.resize(image, (nuevo_w, nuevo_h), interpolation=cv2.INTER_CUBIC)
    
    img_bordered = cv2.copyMakeBorder(img_redim, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    
    config = r'--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789'
    texto = pytesseract.image_to_string(img_bordered, config=config)
    digito = ''.join(filter(str.isdigit, texto.strip()))
    
    return digito if digito else None

class SoccerNetTracker:
    def __init__(self, model_path='yolov8x.pt', conf_threshold=0.3, iou_threshold=0.5, 
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 track_thresh=0.5, track_buffer=30, match_thresh=0.8, frame_rate=25,
                 min_confidence_output=0.8):
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.min_confidence_output = min_confidence_output
        
        print(f"Cargando modelo YOLOv8 desde {model_path}...")
        self.model = YOLO(model_path)
        
        self.person_class = 0
        self.ball_class = 32
        
        print(f"Modelo cargado. Usando dispositivo: {device}")
        print(f"Confianza mínima para salida de tracking: {min_confidence_output}")

    def run_number_recognition_pipeline(self, player_crop):
        if player_crop.size == 0:
            return []

        try:
            torso = crop_torso(player_crop)
            super_res = improve_resolution(torso)
            enhanced = enhance_colors_selectively(super_res)
            simplified1 = simplify_colors_step1(enhanced)
            simplified2 = simplify_colors_step2(simplified1)
            components = isolate_color_components(simplified2)
            
            recognized_digits = []
            for comp_img in components:
                candidate_img = filter_number_candidates(comp_img)
                digit = perform_ocr_on_component(candidate_img)
                if digit:
                    recognized_digits.append(digit)
            
            return list(set(recognized_digits))
        except Exception:
            return []

    def process_sequence(self, sequence_path):
        img_dir = os.path.join(sequence_path, 'img1')
        if not os.path.exists(img_dir):
            print(f"Error: No se encontró la carpeta img1 en {sequence_path}")
            return
        
        images = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        if not images:
            print(f"Error: No se encontraron imágenes JPG en {img_dir}")
            return
        
        det_dir = os.path.join(sequence_path, 'det')
        os.makedirs(det_dir, exist_ok=True)
        output_file = os.path.join(det_dir, "Newdet.txt")
        
        total_images = len(images)
        print(f"Procesando secuencia {os.path.basename(sequence_path)} ({total_images} imágenes)")
        
        all_results = []
        
        progress_iterator = tqdm(images, desc="Iniciando...", total=total_images)
        
        for i, img_file in enumerate(progress_iterator):
            
            progress_iterator.set_description(f"Frame ({i + 1}/{total_images})")
            
            frame_id = int(Path(img_file).stem)
            img_path = os.path.join(img_dir, img_file)
            
            frame_image = cv2.imread(img_path)
            if frame_image is None: continue
                
            track_results = self.model.track(
                source=img_path, conf=self.conf_threshold, iou=self.iou_threshold,
                tracker="bytetrack.yaml", persist=True, verbose=False, device=self.device
            )
            
            if len(track_results) > 0 and hasattr(track_results[0], 'boxes'):
                boxes = track_results[0].boxes
                
                if hasattr(boxes, 'id') and boxes.id is not None:
                    for j in range(len(boxes)):
                        cls = int(boxes.cls[j].item())
                        conf = float(boxes.conf[j].item())
                        
                        if conf < self.min_confidence_output:
                            continue
                        
                        xyxy = boxes.xyxy[j].cpu().numpy()
                        x1, y1, x2, y2 = map(int, xyxy)
                        w, h = x2 - x1, y2 - y1
                        track_id = int(boxes.id[j].item())
                        
                        recognized_numbers = []
                        if cls == self.person_class:
                            player_crop = frame_image[y1:y2, x1:x2]
                            recognized_numbers = self.run_number_recognition_pipeline(player_crop)

                        all_results.append({
                            'frame_id': frame_id,
                            'object_id': -1 if cls == self.ball_class else track_id,
                            'bb_left': x1, 'bb_top': y1, 'bb_width': w, 'bb_height': h,
                            'confidence': conf,
                            'recognized_numbers': recognized_numbers
                        })
        
        self.write_results(all_results, output_file)
        print(f"\nResultados guardados en {output_file}") 

    def write_results(self, results, output_file):
        with open(output_file, 'w') as f:
            for res in results:
                numbers_str = str(res.get('recognized_numbers', []))
                line = (f"{res['frame_id']},{res['object_id']},{res['bb_left']:.2f},"
                        f"{res['bb_top']:.2f},{res['bb_width']:.2f},{res['bb_height']:.2f},"
                        f"{res['confidence']:.6f},-1,-1,-1,{numbers_str}\n")
                f.write(line)

if __name__ == "__main__":

    match_clip = "Ruta/A/La/Carpeta/De/La/Secuencia"
    
    model_path = 'yolov8x.pt'
    conf_threshold = 0.3
    iou_threshold = 0.5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    min_confidence_output = 0.75
    
    tracker = SoccerNetTracker(
        model_path=model_path,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        device=device,
        min_confidence_output=min_confidence_output
    )
    
    tracker.process_sequence(match_clip)

