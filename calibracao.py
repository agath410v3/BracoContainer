import numpy as np
import cv2
import glob

# --- CONFIGURAÇÕES ---
# Número de vértices internos do tabuleiro (largura, altura). 
# Ex: Um tabuleiro de 9x6 quadrados tem 8x5 vértices internos.
CHECKERBOARD = (8, 6)
TAMANHO_QUADRADO_MM = 30.0 # O tamanho que você mediu com a régua!

# Critérios de parada do algoritmo
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepara os pontos 3D no mundo real: (0,0,0), (25,0,0), (50,0,0) ...
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * TAMANHO_QUADRADO_MM

objpoints = [] # Pontos 3D no mundo real
imgpoints = [] # Pontos 2D na imagem (pixels)

# Pega todas as imagens da pasta
images = glob.glob('fotos_calibracao/*.jpg')

print(f"Processando {len(images)} imagens...")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Encontra os cantos do tabuleiro
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret == True:
        print(f"[OK] Padrão encontrado na foto: {fname}")
        objpoints.append(objp)
        
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('Calibrando...', img)
        cv2.waitKey(500) # Deixei 500ms para dar tempo de você ver a tela
    else:
        print(f"[FALHA] O OpenCV não achou o padrão {CHECKERBOARD} na foto: {fname}")

cv2.destroyAllWindows()

# ==========================================
# CÁLCULO FINAL DA CALIBRAÇÃO
# ==========================================
print("Calculando matrizes. Isso pode levar alguns segundos...")
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("\n--- RESULTADOS DA CALIBRAÇÃO ---")
print("Anote estes valores para usar no seu script do xArm:\n")
print("Matriz da Câmera (K):")
print(mtx)
print("\nCoeficientes de Distorção (D):")
print(dist)

# Os parâmetros para o pupil_apriltags são a diagonal principal e o centro:
fx = mtx[0, 0]
fy = mtx[1, 1]
cx = mtx[0, 2]
cy = mtx[1, 2]
print(f"\nValores para o 'camera_params' do AprilTag: [{fx:.2f}, {fy:.2f}, {cx:.2f}, {cy:.2f}]")