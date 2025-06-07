import os
import cv2
from pathlib import Path

def improve_resolution(image):
    """Mejora la resolución usando interpolación bicúbica."""
    height, width = image.shape[:2]
    new_size = (width * 2, height * 2)
    return cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)

def process_image(input_path, output_path):
    image = cv2.imread(input_path)
    if image is None:
        print(f"No se pudo leer la imagen: {input_path}")
        return
    improved = improve_resolution(image)
    cv2.imwrite(output_path, improved)
    print(f"Procesada: {input_path} -> {output_path}")

def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            nombre_sin_extension = Path(filename).stem
            nombre_archivo_png = f"{nombre_sin_extension}.png"
            output_path = os.path.join(output_folder, nombre_archivo_png)
            process_image(input_path, output_path)

input_folder = 'NumRecogFolder/2_0_ImgRecortadas60Central'
output_folder = 'NumRecogFolder/2_1_ImgRecortesSuperResol'
process_folder(input_folder, output_folder)
