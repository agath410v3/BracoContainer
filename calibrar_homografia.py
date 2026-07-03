import time
import numpy as np
import cv2
from pupil_apriltags import Detector
from xarm.wrapper import XArmAPI

# ==========================================
# CONFIGURAÇÕES
# ==========================================
IP_DO_ROBO = '192.168.1.210'
ID_GARRA = 5
INDICE_CAMERA = 0
DELTA_TESTE_MM = 25.0
ARQUIVO_SAIDA = "homografia_place.npy"

# Usamos parâmetros fictícios pois não vamos usar o 3D (pose_t)
camera_params = [600.0, 600.0, 320.0, 240.0]

detector = Detector(families="tag36h11", nthreads=1, quad_decimate=1.0)
cap = cv2.VideoCapture(INDICE_CAMERA, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Conectando ao xArm...")
arm = XArmAPI(IP_DO_ROBO)
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(state=0)
time.sleep(1)

def ler_centro_pixel_tag():
    """Lê APENAS o centro 2D (Pixels X, Y) da tag na imagem."""
    amostras = []
    for _ in range(30): # Tenta ler 30 frames
        ret, frame = cap.read()
        if not ret: continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = detector.detect(gray, estimate_tag_pose=False, camera_params=camera_params, tag_size=0.04)
        
        for r in results:
            if r.tag_id == ID_GARRA:
                amostras.append(r.center)
                break
        time.sleep(0.03)
        
    if len(amostras) < 5:
        raise RuntimeError(f"Não achei a tag {ID_GARRA}!")
        
    return np.median(amostras, axis=0) # Retorna (X_pixel, Y_pixel)

def ir_para_posicao(x, y, z, roll, pitch, yaw):
    arm.set_position(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw, wait=True)
    time.sleep(1.0)

# ==========================================
# ROTINA DO QUADRADO DE CALIBRAÇÃO
# ==========================================
print("\n=== CALIBRANDO POR HOMOGRAFIA (PIXELS -> MM) ===")
input("Pressione ENTER para começar...")

# Pega a posição atual de onde o robô está
code, pos_atual = arm.get_position()
r_x, r_y, r_z, r_roll, r_pitch, r_yaw = pos_atual

pontos_robo_mm = []
pontos_camera_px = []

print("\n[1/4] Ponto de Origem...")
pontos_robo_mm.append([r_x, r_y])
pontos_camera_px.append(ler_centro_pixel_tag())

print(f"[2/4] Movendo X +{DELTA_TESTE_MM}mm...")
ir_para_posicao(r_x + DELTA_TESTE_MM, r_y, r_z, r_roll, r_pitch, r_yaw)
pontos_robo_mm.append([r_x + DELTA_TESTE_MM, r_y])
pontos_camera_px.append(ler_centro_pixel_tag())

print(f"[3/4] Movendo Y +{DELTA_TESTE_MM}mm...")
ir_para_posicao(r_x + DELTA_TESTE_MM, r_y + DELTA_TESTE_MM, r_z, r_roll, r_pitch, r_yaw)
pontos_robo_mm.append([r_x + DELTA_TESTE_MM, r_y + DELTA_TESTE_MM])
pontos_camera_px.append(ler_centro_pixel_tag())

print(f"[4/4] Movendo X -{DELTA_TESTE_MM}mm...")
ir_para_posicao(r_x, r_y + DELTA_TESTE_MM, r_z, r_roll, r_pitch, r_yaw)
pontos_robo_mm.append([r_x, r_y + DELTA_TESTE_MM])
pontos_camera_px.append(ler_centro_pixel_tag())

print("Voltando para origem...")
ir_para_posicao(r_x, r_y, r_z, r_roll, r_pitch, r_yaw)

# ==========================================
# CÁLCULO MÁGICO (Homografia)
# ==========================================
pts_src = np.array(pontos_camera_px, dtype=float) # O que a câmera viu (Pixels)
pts_dst = np.array(pontos_robo_mm, dtype=float)   # Onde o robô estava (Milímetros)

# Encontra a Matriz 3x3 que traduz Pixels perfeitos para Milímetros reais
H, status = cv2.findHomography(pts_src, pts_dst)

print("\n=== MATRIZ DE HOMOGRAFIA (H) GERADA ===")
print(H)
np.save(ARQUIVO_SAIDA, H)
print(f"\n[SUCESSO] Salvo como '{ARQUIVO_SAIDA}'!")

arm.disconnect()
cap.release()