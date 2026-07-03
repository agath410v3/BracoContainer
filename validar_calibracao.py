"""
VALIDAÇÃO DA CALIBRAÇÃO CÂMERA-ROBÔ
=====================================
Testa a precisão da matriz gerada por calibrar_alinhamento.py, medindo o
erro residual real em milímetros.

Como funciona
-------------
Para cada movimento de teste:
  1. Mede a posição da tag da garra na câmera (origem)
  2. Move o robô por um deslocamento CONHECIDO (dx, dy)
  3. Mede a posição da tag na câmera de novo
  4. Usa a matriz de calibração (M) para prever, a partir do deslocamento
     visto pela câmera, qual deveria ter sido o deslocamento do robô
  5. Compara a previsão com o deslocamento real que foi comandado ao robô
  6. A diferença entre os dois é o erro residual da calibração, em mm

Esse é exatamente o tipo de erro que vai aparecer na hora de pegar/soltar
objetos de verdade, então é uma boa estimativa da precisão real do sistema
(diferente de só olhar a matriz e "confiar" nela).

Os movimentos de teste incluem uma direção diagonal (fora dos eixos X/Y
puros usados durante a calibração) para detectar não-linearidades ou
distorção de lente que só aparecem fora dos eixos calibrados.

Requisitos
----------
- Rode calibrar_alinhamento.py antes, para gerar matriz_calibracao_camera_robo.npy
- A tag da garra precisa ficar visível durante todo o teste
- Deixe espaço livre de pelo menos ~40mm em todas as direções ao redor da
  posição atual do robô
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

camera_params = [600.0, 600.0, 320.0, 240.0]
tag_size = 0.04
INDICE_CAMERA = 0

ARQUIVO_MATRIZ = "matriz_calibracao_camera_robo.npy"

# Movimentos de teste (dx, dy) em mm, a partir da posição atual do robô.
# Ajuste os valores conforme o espaço livre disponível na sua mesa.
MOVIMENTOS_TESTE = [
    (40.0, 0.0),
    (0.0, 40.0),
    (-40.0, 0.0),
    (0.0, -40.0),
    (30.0, 30.0),   # diagonal: testa fora dos eixos puros da calibração
]

LIMIAR_OK_MM = 2.0       # erro abaixo disso = ótima precisão
LIMIAR_ALERTA_MM = 5.0   # erro acima disso = recomendado recalibrar

# ==========================================
# SETUP
# ==========================================
try:
    M = np.load(ARQUIVO_MATRIZ)
except FileNotFoundError:
    print(f"[ERRO FATAL] '{ARQUIVO_MATRIZ}' não encontrado.")
    print("Rode calibrar_alinhamento.py primeiro para gerar essa matriz.")
    exit()

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


def capturar_posicao_tag(n_amostras=15):
    """Faz a média de várias detecções da tag da garra para reduzir ruído."""
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
        for r in results:
            if r.tag_id == ID_GARRA and r.pose_t is not None:
                amostras.append((r.pose_t[0][0], r.pose_t[1][0]))
        time.sleep(0.03)

    if len(amostras) < n_amostras // 2:
        raise RuntimeError(
            f"Não consegui detectar a tag {ID_GARRA} de forma confiável. "
            "Verifique iluminação, foco e se a tag ficou visível."
        )

    return np.array(amostras).mean(axis=0)


def mover_robo_delta(dx=0.0, dy=0.0):
    code, pos = arm.get_position()
    if code != 0:
        raise RuntimeError("Erro ao ler posição do robô.")
    x, y, z, roll, pitch, yaw = pos
    arm.set_position(x=x + dx, y=y + dy, z=z, roll=roll, pitch=pitch, yaw=yaw, wait=True)


# ==========================================
# ROTINA DE VALIDAÇÃO
# ==========================================
print("\n=== VALIDAÇÃO DA CALIBRAÇÃO CÂMERA-ROBÔ ===")
print(f"Serão testados {len(MOVIMENTOS_TESTE)} movimentos a partir da posição atual do robô.")
print(f"Garanta que a tag da garra (ID {ID_GARRA}) fique visível e que haja espaço livre.\n")
input("Pressione ENTER para começar...")

erros = []

for i, (dx, dy) in enumerate(MOVIMENTOS_TESTE, start=1):
    print(f"\n[Teste {i}/{len(MOVIMENTOS_TESTE)}] Movimento comandado: dx={dx:+.1f}mm, dy={dy:+.1f}mm")

    cam_antes = capturar_posicao_tag()
    mover_robo_delta(dx=dx, dy=dy)
    time.sleep(0.4)
    cam_depois = capturar_posicao_tag()

    delta_cam = cam_depois - cam_antes
    delta_previsto = M @ delta_cam
    delta_real = np.array([dx, dy])

    erro_vetor = delta_previsto - delta_real
    erro_mm = float(np.linalg.norm(erro_vetor))
    erros.append(erro_mm)

    print(f"  Previsto pela matriz: dx={delta_previsto[0]:+.2f}mm, dy={delta_previsto[1]:+.2f}mm")
    print(f"  Erro residual: {erro_mm:.2f}mm")

    # volta para a origem antes do próximo teste
    mover_robo_delta(dx=-dx, dy=-dy)
    time.sleep(0.6)

erros = np.array(erros)

print("\n=== RESUMO ===")
print(f"Erro médio:  {erros.mean():.2f} mm")
print(f"Erro máximo: {erros.max():.2f} mm")

if erros.max() < LIMIAR_OK_MM:
    print(f"\n[OK] Ótima precisão (erro máximo < {LIMIAR_OK_MM}mm). Calibração validada.")
elif erros.max() < LIMIAR_ALERTA_MM:
    print(f"\n[ALERTA] Precisão aceitável, mas no limite (entre {LIMIAR_OK_MM} e {LIMIAR_ALERTA_MM}mm).")
    print("Considere melhorar iluminação/foco da câmera, ou recalibrar.")
else:
    print(f"\n[ERRO] Erro alto (> {LIMIAR_ALERTA_MM}mm). Recomendado:")
    print("  - Verificar se a tag da garra está bem fixa e plana (sem dobras/inclinação)")
    print("  - Melhorar iluminação e foco da câmera")
    print("  - Rodar calibrar_alinhamento.py de novo")
    print("  - Verificar se a câmera ou o suporte dela se moveu desde a última calibração")

arm.disconnect()
cap.release()
