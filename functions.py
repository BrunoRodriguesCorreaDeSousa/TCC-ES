# Todas as funções do programa estão neste arquivo para manter o arquivo principal limpo.

from ultralytics import YOLO
from datetime import datetime
from time import time
from numpy import random
import cv2
import warnings
import os
with warnings.catch_warnings(action="ignore"):
    warnings.warn('', RuntimeWarning)
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
    from pygame import mixer

# Variáveis globais
seed = random.randint(0, 2 ** 31 - 1)  # Gera uma seed pseudo-aleatória.
prev_frame_time = new_frame_time = gc = 0


# Verifica as classes detectadas e desenha as caixas delimitadoras.
def detect(frame, boxes, labels, score=False):
    global gc
    detected = False
    # Abre um arquivo de log para escrever as detecções.
    logfile = open(f"{datetime.now().date()}.log", "a")
    # Gera as cores para cada uma das classes de forma pseudo-aleatória.
    random.seed(seed)
    colors = random.randint(0, 255, size=(len(labels), 3), dtype="uint8")
    # Loop para iterar sobre cada uma das caixas delimitadoras.
    for box in boxes:
        # Seleciona o rótulo referente à classe da caixa atual e adiciona o nível de confiança ao rótulo caso score=True.
        if int(box[-1]) != 0:  # O último valor do tensor é referente à classe prevista. Verifica se o modelo detectou a classe.
            # Escreve a classe detectada e o horário da detecção no arquivo de log.
            logfile.write(f"{labels[int(box[-1])]} detected at {datetime.now().time()}\n")
            if not detected:
                if gc < 15:
                    gc += 1
                else:
                    detected = True
        elif gc > 0:
            gc = 0
        if score:
            label = labels[int(box[-1])] + " " + str(round(100 * float(box[-2]), 1)) + "%"
        else:
            label = labels[int(box[-1])]
        color = [int(c) for c in colors[int(box[-1])]]  # Seleciona a cor referente à classe da caixa atual.
        bt = max(round(sum(frame.shape) / 2 * 0.003), 2)  # Calcula a espessura das bordas conforme o tamanho do quadro e garante que a espessura mínima seja 2.
        tl, br = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))  # Os quatro primeiros valores do tensor são referentes à posição da caixa. TL é o canto superior esquerdo, BR é o canto inferior direito.
        cv2.rectangle(frame, tl, br, color, thickness=bt, lineType=cv2.LINE_AA)  # Desenha a caixa delimitadora.
        if label:
            ft = max(bt - 1, 1)  # Calcula a grossura da fonte conforme a espessura das bordas e garante que a grossura mínima seja 1.
            w, h = cv2.getTextSize(label, 0, fontScale=bt / 3, thickness=ft)[0]  # Largura e altura do texto.
            outside = tl[1] - h >= 3  # Verifica se há espaço acima da caixa para escrever o texto.
            br = tl[0] + w, tl[1] - h - 3 if outside else tl[1] + h + 3  # Se outside=True desenha o fundo acima da caixa, do contrário desenha abaixo.
            cv2.rectangle(frame, tl, br, color, -1, cv2.LINE_AA)  # Desenha o fundo do rótulo.
            cv2.putText(frame, label, (tl[0], tl[1] - 2 if outside else tl[1] + h + 2), 0, bt / 3,
                        (255, 255, 255),
                        thickness=ft, lineType=cv2.LINE_AA)  # Escreve o texto com o rótulo.
    logfile.close()
    return detected


# Redimensiona uma imagem.
def resize(w, h, res):
    if h != res:
        ratio = w / h
        height = res
        width = int(height * ratio)
        return width, height
    return w, h


# Mostra a quantidade de quadros por segundo.
def show_fps(frame):
    global new_frame_time, prev_frame_time
    new_frame_time = time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps = str(fps)
    cv2.putText(frame, fps, (10, 30), 0, 1, (100, 255, 0), 3, cv2.LINE_AA)
    return frame


# Executa o modelo.
def run_model(video, window):
    global new_frame_time, prev_frame_time, gc
    prev_frame_time = new_frame_time = gc = 0
    model = YOLO("runs/detect/crime/weights/last.pt")
    v = video.read()[1]
    w, h = resize(v.shape[1], v.shape[0], 640)
    mixer.init()
    mixer.music.load("sound/alarm.mp3")
    playing = False
    window.withdraw()
    while video.isOpened():
        if cv2.waitKey(1) == 27:
            break
        on, frame = video.read()
        if not on:
            break
        frame = cv2.resize(frame, (w, h))
        results = model(frame, half=True, verbose=False)
        if detect(frame, results[0].boxes.data, results[0].names) and not playing:
            mixer.music.play()  # Toca um arquivo de áudio para simbolizar o disparo do alarme.
            playing = True
        show_fps(frame)
        cv2.imshow("Video", frame)
    mixer.music.stop()
    mixer.music.unload()
    window.deiconify()
    cv2.destroyAllWindows()
    return
