import os
import subprocess
import glob

def procesar_keypoints_a_txt(input_folder, output_txt_path):
    image_files = sorted(glob.glob(os.path.join(input_folder, "*.jpg")))
    if not image_files:
        print(f"No hay imágenes .jpg en {input_folder}")
        return

    all_lines = []

    for img_path in image_files:
        img_name = os.path.basename(img_path)
        frame_id, _ = os.path.splitext(img_name)
        cmd = [
            "python", "PnLCalib/ExtraeCoordPuntos.py",
            "--weights_kp", "SV_kp",
            "--weights_line", "SV_lines",
            "--pnl_refine",
            "--device", "cpu",
            "--input_path", img_path,
            "--frame_id", frame_id
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error procesando {img_name}: {result.stderr}")
            continue

        lines = result.stdout.strip().split('\n')
        for l in lines:
            if l.strip():
                all_lines.append(l.strip())

    with open(output_txt_path, "w") as of:
        of.write("\n".join(all_lines))
    print(f"Keypoints por frame guardados en {output_txt_path}")

if __name__ == '__main__':
    #CORRER DESDE AQUI
    input_images_path = "Ruta/A/La/Carpeta/De/La/Secuencia"
    output_txt_path = "Ruta/A/La/Carpeta/De/Salida"
    procesar_keypoints_a_txt(input_images_path, output_txt_path)
