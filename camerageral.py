import cv2
import time
import numpy as np
from pupil_apriltags import Detector
from xarm.wrapper import XArmAPI

# ==========================================
# 1. CONFIGURAÇÕES DOS IDs E DO ROBÔ
# ==========================================
IP_DO_ROBO = '192.168.1.210'

ID_DESTINO = 6
ID_CUBOS_MINIMO = 7 # Qualquer AprilTag >= 7 será considerada um objeto!

# ==========================================
# 2. SINTONIA FINA, ALTURAS E ESPAÇAMENTO
# ==========================================
ALTURA_PEGADA_Z = 175.0  
ALTURA_SOLTURA_Z = 50.0 

COMPENSACAO_X_MM = -10.0 
COMPENSACAO_Y_MM = 30.0 

# Distância que o robô vai dar entre um cubo e outro no Destino
ESPACO_ENTRE_CUBOS_MM = 55.0  

print("Conectando ao xArm...")
arm = XArmAPI(IP_DO_ROBO)
arm.clean_error() 
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(state=0)

print("Habilitando a garra...")
arm.set_gripper_enable(True)
arm.set_gripper_mode(0)
arm.set_gripper_speed(3000)
time.sleep(1)

POS_HOME = [300.0, 0.0, 250.0, 180.0, 0.0, 0.0]

print(">> Movendo para a posição inicial de segurança...")
arm.set_gripper_position(800, wait=True) 
time.sleep(0.5)
arm.set_position(*POS_HOME, wait=True)

# ==========================================
# 3. CARREGAR AS DUAS HOMOGRAFIAS (PICK E PLACE)
# ==========================================
try:
    H_MATRIX_PICK = np.load("homografia_pick.npy")
    H_MATRIX_PLACE = np.load("homografia_place.npy")
    print(">> Matrizes Multi-Plano carregadas com sucesso!")
except FileNotFoundError:
    print("\n[ERRO FATAL] Rode os scripts de calibração para Pick e Place primeiro.")
    arm.disconnect()
    exit()

def converter_pixel_para_mm(px_x, px_y, matriz_homografia):
    ponto_pixel = np.array([[[px_x, px_y]]], dtype=np.float32)
    ponto_mm = cv2.perspectiveTransform(ponto_pixel, matriz_homografia)
    return ponto_mm[0][0][0], ponto_mm[0][0][1]

# ==========================================
# 4. CONFIGURAÇÃO DA CÂMERA
# ==========================================
detector = Detector(families="tag36h11", nthreads=1, quad_decimate=1.0)

INDICE_CAMERA = 0 
cap = cv2.VideoCapture(INDICE_CAMERA, cv2.CAP_V4L2)

if not cap.isOpened():
    print(f"\n[ERRO FATAL] A câmera {INDICE_CAMERA} não abriu.")
    arm.disconnect()
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
time.sleep(1.0)

cv2.namedWindow("Linha de Producao", cv2.WINDOW_NORMAL)
print("Sistema aguardando. Pressione 'q' para sair.")

# ==========================================
# 5. LOOP DE MONITORAÇÃO
# ==========================================
while True:
    ret, frame = cap.read()
    if not ret or frame is None: continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = detector.detect(gray, estimate_tag_pose=False)

    tags_detectadas = {} 
    pecas_encontradas = []

    for r in results:
        corners = r.corners.astype(int)
        for i in range(4): cv2.line(frame, tuple(corners[i]), tuple(corners[(i + 1) % 4]), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {r.tag_id}", tuple(corners[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        tags_detectadas[r.tag_id] = r
        if r.tag_id >= ID_CUBOS_MINIMO:
            pecas_encontradas.append(r)

    sistema_pronto = (len(pecas_encontradas) > 0) and (ID_DESTINO in tags_detectadas)

    if sistema_pronto:
        texto = f"PRONTO: {len(pecas_encontradas)} cubo(s) achados! Aperte ESPACO!"
        cv2.putText(frame, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Linha de Producao", frame)

    tecla = cv2.waitKey(1) & 0xFF
    if tecla == ord("q"): break
        
    elif tecla == 32 and sistema_pronto:
        print("\n[!] Iniciando Limpeza da Mesa...")
        cv2.waitKey(1)
        
        # 1. MEMORIZA O DESTINO
        px_dst_x, px_dst_y = tags_detectadas[ID_DESTINO].center
        base_place_x, base_place_y = converter_pixel_para_mm(px_dst_x, px_dst_y, H_MATRIX_PLACE)
        alvo_place_base_x = base_place_x + COMPENSACAO_X_MM
        alvo_place_base_y = base_place_y + COMPENSACAO_Y_MM
        
        cubos_movidos = 0
        tentativas_vazias = 0
        
        # O CADERNINHO DE MEMÓRIA DO ROBÔ
        cubos_ja_processados = [] 
        
        # 2. LOOP AUTOMÁTICO DE TRABALHO
        while True:
            # Espera o braço parar de balançar
            time.sleep(0.5) 
            
            # Esvazia agressivamente as fotos velhas do buffer
            for _ in range(20): cap.grab() 
            
            ret, frame_auto = cap.read() # Pega uma foto fresca do "agora"
            if not ret: break
            
            gray_auto = cv2.cvtColor(frame_auto, cv2.COLOR_BGR2GRAY)
            results_auto = detector.detect(gray_auto, estimate_tag_pose=False)
            
            # ==========================================
            # O FILTRO MÁGICO: Só aceita >= 7 E que NÃO esteja na lista de memória!
            # ==========================================
            cubos_restantes = [
                r for r in results_auto 
                if (r.tag_id >= ID_CUBOS_MINIMO) and (r.tag_id not in cubos_ja_processados)
            ]
            
            if len(cubos_restantes) == 0:
                tentativas_vazias += 1
                print(f"-> Nenhuma peça NOVA detectada (Tentativa {tentativas_vazias}/3)...")
                if tentativas_vazias >= 3:
                    print("\n>> Mesa limpa com sucesso! Aguardando nova remessa...")
                    break # Sai do ciclo automático e volta a pedir ESPACO
                continue 
            
            tentativas_vazias = 0
                
            cubos_restantes.sort(key=lambda x: x.tag_id)
            cubo_alvo = cubos_restantes[0]
            print(f"\n-> Cubo alvo atual: ID {cubo_alvo.tag_id}")
            
            px_obj_x, px_obj_y = cubo_alvo.center
            
            base_pick_x, base_pick_y = converter_pixel_para_mm(px_obj_x, px_obj_y, H_MATRIX_PICK)
            alvo_pick_x = base_pick_x + COMPENSACAO_X_MM
            alvo_pick_y = base_pick_y + COMPENSACAO_Y_MM
            
            alvo_place_x = alvo_place_base_x + (cubos_movidos * ESPACO_ENTRE_CUBOS_MM)
            alvo_place_y = alvo_place_base_y - (cubos_movidos * ESPACO_ENTRE_CUBOS_MM)
            
            code, _ = arm.get_position()
            if code == 0:
                arm.set_position(x=alvo_pick_x, y=alvo_pick_y, z=ALTURA_PEGADA_Z + 150.0, roll=180, pitch=0, yaw=0, wait=True)
                arm.set_position(x=alvo_pick_x, y=alvo_pick_y, z=ALTURA_PEGADA_Z, roll=180, pitch=0, yaw=0, wait=True)
                
                arm.set_gripper_position(200, wait=True) 
                time.sleep(0.5)
                arm.set_position(x=alvo_pick_x, y=alvo_pick_y, z=ALTURA_PEGADA_Z + 150.0, roll=180, pitch=0, yaw=0, wait=True)
                
                print(f"-> Levando para a posição na fila...")
                arm.set_position(x=alvo_place_x, y=alvo_place_y, z=ALTURA_SOLTURA_Z + 150.0, roll=180, pitch=0, yaw=0, wait=True)
                arm.set_position(x=alvo_place_x, y=alvo_place_y, z=ALTURA_SOLTURA_Z, roll=180, pitch=0, yaw=0, wait=True)
                
                arm.set_gripper_position(800, wait=True)
                time.sleep(0.5)
                arm.set_position(x=alvo_place_x, y=alvo_place_y, z=ALTURA_SOLTURA_Z + 150.0, roll=180, pitch=0, yaw=0, wait=True)
                
                # ==========================================
                # SUCESSO! Anota a ID na memória para ignorar na próxima
                # ==========================================
                cubos_ja_processados.append(cubo_alvo.tag_id)
                cubos_movidos += 1
                
                print("-> Retornando para home e analisando a mesa...")
                arm.set_position(*POS_HOME, wait=True)
            else:
                print("Erro no braço! Abortando limpeza.")
                arm.clean_error()
                break

arm.disconnect()
cap.release()
cv2.destroyAllWindows()