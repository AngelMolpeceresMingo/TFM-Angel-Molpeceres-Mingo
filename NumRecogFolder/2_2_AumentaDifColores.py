import os
import cv2
import numpy as np
from pathlib import Path

def enhance_colors_selectively(image, saturation_boost=2.0, contrast_alpha=1.3, gray_threshold=30):
    # Trabajar en espacio HSV para saturación
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Trabajar en espacio LAB para luminosidad
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # 1. Mejorar contraste en luminosidad (LAB)
    l_enhanced = cv2.convertScaleAbs(l, alpha=contrast_alpha, beta=0)
    
    # 2. Aplicar CLAHE para mejor distribución de luminosidad
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_enhanced = clahe.apply(l_enhanced)
    
    # 3. Crear máscara para áreas con color (corregido)
    # Convertir a y b a int16 para manejar valores negativos correctamente
    a_int = a.astype(np.int16) - 128
    b_int = b.astype(np.int16) - 128
    color_intensity = np.sqrt(a_int**2 + b_int**2).astype(np.uint8)
    _, color_mask = cv2.threshold(color_intensity, gray_threshold, 255, cv2.THRESH_BINARY)
    
    # 4. Aumentar saturación selectivamente
    mask_float = (color_mask / 255.0).astype(np.float32)
    s_float = s.astype(np.float32)
    s_enhanced = s_float * (1.0 + saturation_boost * mask_float)
    s_enhanced = np.clip(s_enhanced, 0, 255).astype(np.uint8)
    
    # 5. Reconstruir imágenes
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    hsv_enhanced = cv2.merge([h, s_enhanced, v])
    
    # 6. Convertir ambas a BGR para mezcla final
    result_lab = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    result_hsv = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
    
    # 7. Mezcla final coherente
    return cv2.addWeighted(result_hsv, 0.6, result_lab, 0.4, 0)

def process_image(input_path, output_path):
    """Procesa una imagen individual y guarda el resultado."""
    image = cv2.imread(input_path)
    if image is None:
        print(f"No se pudo leer la imagen: {input_path}")
        return False
    
    try:
        result = enhance_colors_selectively(image)
        success = cv2.imwrite(output_path, result)
        if success:
            print(f"Procesada: {input_path} -> {output_path}")
            return True
        else:
            print(f"Error al guardar: {output_path}")
            return False
    except Exception as e:
        print(f"Error procesando {input_path}: {str(e)}")
        return False

def process_folder(input_folder, output_folder):
    """Procesa todas las imágenes en una carpeta."""
    if not os.path.exists(input_folder):
        print(f"La carpeta de entrada no existe: {input_folder}")
        return
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    processed_count = 0
    total_count = 0
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            total_count += 1
            input_path = os.path.join(input_folder, filename)
            
            nombre_sin_extension = Path(filename).stem
            output_path = os.path.join(output_folder, f"{nombre_sin_extension}.png")
            
            if process_image(input_path, output_path):
                processed_count += 1
    
    print(f"Procesamiento completado: {processed_count}/{total_count} imágenes")

# Ejecutar
input_folder = 'NumRecogFolder/2_1_ImgRecortesSuperResol'
output_folder = 'NumRecogFolder/2_2_ImgColoresExagerados'

process_folder(input_folder, output_folder)
