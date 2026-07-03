import cv2
import os

# Cria a pasta se ela não existir
if not os.path.exists('fotos_calibracao'):
    os.makedirs('fotos_calibracao')

# Ajuste para o ID da sua câmera (0, 1, etc)
cap = cv2.VideoCapture(1)

# FORÇANDO A RESOLUÇÃO (Mude para a resolução que você vai usar no projeto final)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

contador = 0
print("Pressione ESPAÇO para tirar a foto. Pressione 'q' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Falha ao capturar a câmera.")
        break

    cv2.imshow("Captura de Calibracao", frame)
    
    key = cv2.waitKey(1)
    
    # Se apertar a BARRA DE ESPAÇO (código 32)
    if key == 32: 
        nome_arquivo = f"fotos_calibracao/tabuleiro_{contador:02d}.jpg"
        cv2.imwrite(nome_arquivo, frame)
        print(f"Foto salva: {nome_arquivo} | Total: {contador + 1}")
        contador += 1
        
    # Se apertar 'q'
    elif key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Captura finalizada!")