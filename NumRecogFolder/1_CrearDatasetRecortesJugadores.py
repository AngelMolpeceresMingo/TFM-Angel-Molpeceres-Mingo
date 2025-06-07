import os
import cv2
import numpy as np
import random
from ultralytics import YOLO
from pathlib import Path

def create_player_detection_dataset(
    images_folder, 
    output_dir="NumRecogFolder", 
    num_samples=20, 
    confidence_threshold=0.5
):
    """
    Crea un dataset de recortes de jugadores a partir de imágenes de fútbol
    
    Args:
        images_folder: Carpeta con imágenes de partidos
        output_dir: Directorio donde guardar los recortes
        num_samples: Número de recortes a extraer
        confidence_threshold: Umbral de confianza para las detecciones
    """
    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Listar todas las imágenes en el directorio y subdirectorios
    image_files = []
    for root, _, files in os.walk(images_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))
    
    print(f"Encontradas {len(image_files)} imágenes en {images_folder}")
    
    # Seleccionar imágenes aleatorias
    if len(image_files) == 0:
        print(f"No se encontraron imágenes en {images_folder}")
        return
    
    selected_images = random.sample(image_files, min(len(image_files), num_samples * 2))
    
    # Cargar modelo YOLOv8 para detección de personas
    model = YOLO('yolov8x.pt')
    
    # Contador de recortes guardados
    saved_crops = 0
    
    for img_path in selected_images:
        if saved_crops >= num_samples:
            break
            
        # Cargar imagen
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"Error al leer la imagen {img_path}")
            continue
        
        # Convertir a RGB para procesamiento
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detectar personas
        results = model(image_rgb)
        
        # Filtrar solo detecciones de personas
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                if box.cls[0] == 0:  # Clase 0 = persona
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    
                    if confidence > confidence_threshold:
                        detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': confidence
                        })
        
        # Si no hay detecciones, continuar con la siguiente imagen
        if not detections:
            continue
        
        # Seleccionar una detección aleatoria
        detection = random.choice(detections)
        x1, y1, x2, y2 = detection['bbox']
        
        # Extraer el recorte del jugador
        player_crop = image[y1:y2, x1:x2]
        
        # Guardar el recorte
        img_filename = os.path.basename(img_path)
        crop_filename = os.path.join(output_dir, f"player_{saved_crops:02d}_{img_filename}")
        cv2.imwrite(crop_filename, player_crop)
        
        print(f"Guardado recorte {saved_crops+1}/{num_samples}: {crop_filename}")
        saved_crops += 1
    
    print(f"Dataset creado con {saved_crops} recortes de jugadores en {output_dir}")

# Ejemplo de uso
if __name__ == "__main__":
    # Ruta a la carpeta con imágenes de partidos
    images_folder = "SoccerNetData/tracking-2023/challenge2023/challenge2023"
    
    # Crear el dataset
    create_player_detection_dataset(
        images_folder=images_folder,
        output_dir="NumRecogFolder",
        num_samples=20,
        confidence_threshold=0.5
    )
