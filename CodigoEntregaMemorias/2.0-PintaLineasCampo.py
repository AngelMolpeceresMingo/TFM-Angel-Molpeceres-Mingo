import os
import subprocess
import glob
from tqdm import tqdm

def procesar_carpeta_con_pnlcalib(input_folder, output_folder):

    os.makedirs(output_folder, exist_ok=True)
    image_files = glob.glob(os.path.join(input_folder, "*.jpg"))
    print(f"Procesando {len(image_files)} imágenes...")
    
    for i, image_path in enumerate(tqdm(image_files, desc="Procesando imágenes")):
        image_name = os.path.basename(image_path)
        abs_image_path = os.path.abspath(image_path)
        abs_output_path = os.path.abspath(os.path.join(output_folder, f"result_{image_name}"))
        
        cmd = [
            "python", "PnLCalib/BestInferenceColours.py",
            "--weights_kp", "SV_kp",
            "--weights_line", "SV_lines", 
            "--pnl_refine",
            "--device", "cpu",
            "--input_path", abs_image_path,
            "--input_type", "image",
            "--save_path", abs_output_path
        ]
        
        subprocess.run(cmd)

if __name__ == '__main__':
    
    input_path = "Ruta/A/La/Carpeta/De/La/Secuencia"
    output_path = "Ruta/A/La/Carpeta/De/La/Secuencia/De/Salida"

    procesar_carpeta_con_pnlcalib(input_path, output_path)
