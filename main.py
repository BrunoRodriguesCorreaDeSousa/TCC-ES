# UniCesumar
# Trabalho de Conclusão de Curso
# Sistema de Visão Computacional para Detecção de Objetos e Reconhecimento de Pessoas
# Desenvolvido por Bruno Rodrigues Correa de Sousa
# RA 22141448-5

from functions import *
import multiprocessing
import customtkinter


# Função principal. Contém o código da interface do programa. Outras funções em functions.py
def main():
    files = []
    for file in os.listdir("videos"):
        if file.endswith(".mp4"):
            files.append(os.path.join("videos", file))

    customtkinter.set_appearance_mode("dark")
    customtkinter.set_default_color_theme("dark-blue")
    window = customtkinter.CTk()
    window.title("Computer Vision")

    source = customtkinter.StringVar(value='f')

    cb = customtkinter.CTkComboBox(window, values=files, justify="left")
    cb.set(files[0])

    def cbg():
        cb.grid(sticky=customtkinter.W, column=1, row=1)

    cbg()

    customtkinter.CTkLabel(window, text="Select the video source:", justify="left", anchor="w").grid(
        sticky=customtkinter.W, column=0, row=0)
    customtkinter.CTkRadioButton(window, text="File", value='f', variable=source, command=cbg).grid(
        sticky=customtkinter.W, column=0, row=1)
    customtkinter.CTkRadioButton(window, text="Camera", value='c', variable=source, command=cb.grid_forget).grid(
        sticky=customtkinter.W, column=0, row=2)

    def run():
        video = cv2.VideoCapture(files[0], cv2.CAP_ANY, (cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY))
        match source.get():
            case 'f':
                video = cv2.VideoCapture(cb.get(), cv2.CAP_ANY,
                                         (cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY))
            case 'c':
                video = cv2.VideoCapture(0, cv2.CAP_ANY,
                                         (cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY))
        run_model(video, window)
        video.release()

    customtkinter.CTkButton(window, text="Run", command=run).grid(sticky=customtkinter.W, column=1, row=3)
    window.mainloop()


if __name__ == "__main__":
    multiprocessing.freeze_support()  # Habilita o suporte à multiprocessamento.
    main()  # Chama a função principal.
