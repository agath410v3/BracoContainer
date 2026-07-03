"""
CALIBRAÇÃO DE ALINHAMENTO CÂMERA-ROBÔ
=======================================
Esse script descobre a matriz 2x2 que converte um deslocamento medido
pela câmera (em metros, a partir da pose do AprilTag da garra) em um
deslocamento correto no frame do robô (em mm).

Por quê isso é necessário?
---------------------------
O script original (camera.py) assumia que os eixos X/Y da câmera são
exatamente paralelos aos eixos X/Y do robô, e multiplicava o delta por
1000.0 (m -> mm). Se a câmera estiver rotacionada (mesmo levemente) em
relação à base do xArm, essa suposição falha e o robô se move na
direção errada. Esse script mede a rotação e a escala reais e gera uma
matriz de correção.

Como funciona
-------------
1. Mede a posição da tag da garra na câmera (origem)
2. Move o robô +DELTA_TESTE_MM em X, mede a tag de novo
3. Volta, move o robô +DELTA_TESTE_MM em Y, mede a tag de novo
4. Resolve o sistema linear para achar a matriz M tal que:
       delta_robot_mm = M @ delta_camera_m

Requisitos antes de rodar
--------------------------
- A tag da garra (ID_GARRA) precisa estar visível pela câmera durante
  TODO o processo, incluindo durante os movimentos.
- Deixe uma área livre de pelo menos DELTA_TESTE_MM x DELTA_TESTE_MM
  em torno da posição atual do robô, pois ele vai se mover de fato.
- Rode esse script de novo sempre que mexer a câmera ou o robô de lugar.
"""

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

DELTA_TESTE_MM = 80.0  # distância de teste em cada eixo (ajuste conforme espaço livre)
# Quanto maior esse valor, menor o impacto relativo do ruído de detecção da
# câmera no resultado final. Use o maior valor que o espaço livre permitir.

camera_params = [600.0, 600.0, 320.0, 240.0]
tag_size = 0.04

INDICE_CAMERA = 0

ARQUIVO_SAIDA = "matriz_calibracao_camera_robo.npy"

# ==========================================
# SETUP
# ==========================================
detector = Detector(families="tag36h11", nthreads=1, quad_decimate=1.0)

cap = cv2.VideoCapture(INDICE_CAMERA, cv2.CAP_V4L2)
if not cap.isOpened():
    print(f"[ERRO FATAL] A câmera {INDICE_CAMERA} não abriu.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
time.sleep(1.0)

print("Conectando ao xArm...")
arm = XArmAPI(IP_DO_ROBO)
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(state=0)
time.sleep(1)


def capturar_posicao_tag(n_amostras=20):
    """Faz a leitura robusta da tag da garra: coleta várias amostras, descarta
    outliers (frames com leitura muito diferente da mediana — geralmente causados
    por blur de movimento ou detecção momentaneamente ruim) e retorna a mediana
    das amostras restantes."""
    amostras = []
    tentativas = 0
    while len(amostras) < n_amostras and tentativas < n_amostras * 15:
        tentativas += 1
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = detector.detect(
            gray, estimate_tag_pose=True,
            camera_params=camera_params, tag_size=tag_size
        )
        tags_garra = [r for r in results if r.tag_id == ID_GARRA and r.pose_t is not None]
        if len(tags_garra) > 1:
            print(f"  [AVISO] {len(tags_garra)} tags com ID {ID_GARRA} detectadas no mesmo frame "
                  "— descartando essa leitura (verifique se não há tag duplicada na cena).")
            continue
        if len(tags_garra) == 1:
            r = tags_garra[0]
            amostras.append((r.pose_t[0][0], r.pose_t[1][0]))
        time.sleep(0.03)

    if len(amostras) < n_amostras // 2:
        raise RuntimeError(
            f"Não consegui detectar a tag {ID_GARRA} de forma confiável "
            f"(só {len(amostras)}/{n_amostras} leituras válidas). "
            "Verifique iluminação, foco e se a tag ficou visível."
        )

    amostras = np.array(amostras)
    mediana = np.median(amostras, axis=0)

    # Rejeita outliers: descarta amostras a mais de 3x o desvio absoluto
    # mediano (MAD) de distância da mediana.
    distancias = np.linalg.norm(amostras - mediana, axis=1)
    mad = np.median(distancias) + 1e-9
    boas = amostras[distancias < 8 * mad]
    if len(boas) < len(amostras):
        print(f"  [INFO] {len(amostras) - len(boas)} amostra(s) descartada(s) como outlier.")

    media_final = boas.mean(axis=0)
    desvio = boas.std(axis=0)
    print(f"  -> posição: ({media_final[0]:.4f}, {media_final[1]:.4f}) m | "
          f"desvio: ({desvio[0]:.4f}, {desvio[1]:.4f}) m | amostras válidas: {len(boas)}")
    return media_final


def mover_robo_delta(dx=0.0, dy=0.0):
    code, pos = arm.get_position()
    if code != 0:
        raise RuntimeError("Erro ao ler posição do robô.")
    x, y, z, roll, pitch, yaw = pos
    arm.set_position(x=x + dx, y=y + dy, z=z, roll=roll, pitch=pitch, yaw=yaw, wait=True)


# ==========================================
# ROTINA DE CALIBRAÇÃO
# ==========================================
print("\n=== CALIBRAÇÃO DE ALINHAMENTO CÂMERA-ROBÔ ===")
print(f"O robô vai se mover {DELTA_TESTE_MM}mm em X e depois em Y, e voltar.")
print(f"Garanta que a tag da garra (ID {ID_GARRA}) fique visível durante todo o processo.\n")
input("Pressione ENTER para começar...")

print("\n[1/3] Capturando posição de referência (origem)...")
time.sleep(0.5)
cam_origem = capturar_posicao_tag()

print(f"\n[2/3] Movendo robô +{DELTA_TESTE_MM}mm em X...")
mover_robo_delta(dx=DELTA_TESTE_MM)
time.sleep(1.0)  # assentamento: evita ler a tag enquanto o braço ainda vibra
cam_apos_x = capturar_posicao_tag()

print("\n>> Voltando para a origem em X...")
mover_robo_delta(dx=-DELTA_TESTE_MM)
time.sleep(1.2)

print(f"\n[3/3] Movendo robô +{DELTA_TESTE_MM}mm em Y...")
mover_robo_delta(dy=DELTA_TESTE_MM)
time.sleep(1.0)
cam_apos_y = capturar_posicao_tag()

print("\n>> Voltando para a origem em Y...")
mover_robo_delta(dy=-DELTA_TESTE_MM)
time.sleep(1.2)

# ==========================================
# CÁLCULO DA MATRIZ DE TRANSFORMAÇÃO
# ==========================================
vetor_u = cam_apos_x - cam_origem   # deslocamento na câmera correspondente a (+DELTA, 0) no robô
vetor_v = cam_apos_y - cam_origem   # deslocamento na câmera correspondente a (0, +DELTA) no robô

# ------------------------------------------------------------------
# DIAGNÓSTICOS — verificam as causas mais comuns de erro grande
# ------------------------------------------------------------------
print("\n=== DIAGNÓSTICO ===")

mag_u_mm = np.linalg.norm(vetor_u) * 1000.0
mag_v_mm = np.linalg.norm(vetor_v) * 1000.0
print(f"Deslocamento visto pela câmera ao mover {DELTA_TESTE_MM}mm em X: {mag_u_mm:.1f}mm equivalente")
print(f"Deslocamento visto pela câmera ao mover {DELTA_TESTE_MM}mm em Y: {mag_v_mm:.1f}mm equivalente")

problemas = []

# 1) A câmera viu praticamente nenhum movimento -> o robô não moveu de fato,
#    a tag travou numa leitura antiga, ou o buffer da câmera está atrasado.
if mag_u_mm < DELTA_TESTE_MM * 0.3 or mag_v_mm < DELTA_TESTE_MM * 0.3:
    problemas.append(
        "A câmera detectou um deslocamento muito menor do que o esperado em pelo "
        "menos um eixo. Isso sugere que o robô não se moveu o esperado, a tag não "
        "estava visível durante o movimento, ou frames antigos (buffer da câmera) "
        "estão sendo lidos."
    )

# 2) A câmera viu um deslocamento muito maior do que o esperado -> camera_params
#    (intrínsecos) provavelmente muito errados para essa distância/altura.
if mag_u_mm > DELTA_TESTE_MM * 3 or mag_v_mm > DELTA_TESTE_MM * 3:
    problemas.append(
        "A câmera detectou um deslocamento muito maior do que o esperado em pelo "
        "menos um eixo. Isso é sinal de que os 'camera_params' (fx, fy, cx, cy) "
        "estão bem errados para a altura/distância atual da câmera."
    )

# 3) Os dois vetores são quase colineares (ângulo perto de 0° ou 180°) -> matriz
#    numericamente instável, qualquer ruído pequeno gera erro enorme na inversão.
cos_ang = np.dot(vetor_u, vetor_v) / (np.linalg.norm(vetor_u) * np.linalg.norm(vetor_v) + 1e-12)
ang = np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0)))
print(f"Ângulo entre os vetores medidos (X vs Y): {ang:.1f}° (ideal: perto de 90°)")
if ang < 20 or ang > 160:
    problemas.append(
        f"O ângulo entre os deslocamentos medidos em X e Y foi de {ang:.1f}°, muito "
        "longe dos 90° esperados. Isso torna a matriz numericamente instável — "
        "pequenos ruídos de detecção geram erros grandes depois da inversão. "
        "Provavelmente um dos dois movimentos não foi capturado corretamente."
    )

if problemas:
    print("\n[ATENÇÃO] Possíveis causas do erro identificadas nesta calibração:")
    for p in problemas:
        print(f"  - {p}")
    print(
        "\nRecomendado corrigir o problema acima ANTES de confiar na matriz gerada "
        "agora. Mesmo assim, a matriz será salva — você pode rodar validar_calibracao.py "
        "para confirmar se o problema realmente afetou o resultado."
    )
else:
    print("Nenhum problema óbvio detectado nos vetores de calibração. Bom sinal.")


C = np.column_stack([vetor_u, vetor_v])  # matriz 2x2 [u | v]

if abs(np.linalg.det(C)) < 1e-9:
    raise RuntimeError(
        "Matriz singular — os deslocamentos medidos na câmera foram nulos ou "
        "colineares. Verifique se o robô realmente se moveu e se a tag ficou "
        "visível em todas as capturas."
    )

# Queremos M tal que M @ C = DELTA_TESTE_MM * I  =>  M = DELTA_TESTE_MM * inv(C)
M = DELTA_TESTE_MM * np.linalg.inv(C)

print("\n=== RESULTADO ===")
print("Matriz de transformação câmera -> robô (delta_robot_mm = M @ delta_camera_m):")
print(M)

escala_x = np.linalg.norm(M[:, 0])
escala_y = np.linalg.norm(M[:, 1])
print(f"\nEscala efetiva: ~{escala_x:.1f} mm/m (eixo X-câmera), ~{escala_y:.1f} mm/m (eixo Y-câmera)")
print("(o script original usava um fator fixo de 1000.0 mm/m, assumindo eixos perfeitamente alinhados)")

angulo_rotacao = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
print(f"Rotação estimada entre os eixos da câmera e do robô: ~{angulo_rotacao:.1f}°")

if abs(angulo_rotacao) > 5 or abs(angulo_rotacao - 180) < 5 or abs(angulo_rotacao + 180) < 5:
    print("\n[ATENÇÃO] A rotação detectada é significativa. Isso confirma que a suposição")
    print("de alinhamento perfeito do script original estava introduzindo erro de posição.")
else:
    print("\nA rotação é pequena, mas ainda assim usar a matriz calibrada é mais preciso")
    print("do que o fator fixo de escala usado antes.")

np.save(ARQUIVO_SAIDA, M)
print(f"\n>> Matriz salva em '{ARQUIVO_SAIDA}'")
print(">> O camera.py atualizado já carrega esse arquivo automaticamente.")
print(">> Recomendado: rode essa calibração de novo sempre que mover a câmera ou o robô.")

arm.disconnect()
cap.release()