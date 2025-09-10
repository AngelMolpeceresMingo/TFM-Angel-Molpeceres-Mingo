import os
import subprocess
import glob
from tqdm import tqdm

def procesar_carpeta_keypoints_from_lines(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    image_files = glob.glob(os.path.join(input_folder, "*.jpg"))
    if not image_files:
        print(f"No se encontraron imágenes .jpg en: {input_folder}")
        return
    
    print(f"Procesando {len(image_files)} imágenes para calcular puntos desde líneas...")
    
    success_count = 0
    error_count = 0
    
    for i, image_path in enumerate(tqdm(image_files, desc="Calculando puntos desde líneas")):
        image_name = os.path.basename(image_path)
        abs_image_path = os.path.abspath(image_path)
        abs_output_path = os.path.abspath(os.path.join(output_folder, f"keypoints_lines_{image_name}"))
        
        cmd = [
            "python", "PnLCalib/points_inference.py",
            "--weights_kp", "SV_kp",
            "--weights_line", "SV_lines", 
            "--pnl_refine",
            "--device", "cpu",
            "--input_path", abs_image_path,
            "--save_path", abs_output_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                success_count += 1
            else:
                error_count += 1
                print(f"Error procesando {image_name}: {result.stderr}")
        except Exception as e:
            error_count += 1
            print(f"Excepción procesando {image_name}: {e}")
    
    print(f"\nRESUMEN DEL PROCESAMIENTO:")
    print(f"Imágenes procesadas exitosamente: {success_count}")
    print(f"Imágenes con errores: {error_count}")
    print(f"Resultados guardados en: {output_folder}")
    


if __name__ == '__main__':

    input_images_path = "Ruta/A/La/Carpeta/De/La/Secuencia"
    output_images_path = "Ruta/A/La/Carpeta/De/La/Secuencia/De/Salida"
    
    procesar_carpeta_keypoints_from_lines(input_images_path, output_images_path)