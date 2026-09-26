import sys
import os

def beep():
    """Emite um beep simples (cross-platform)."""
    if sys.platform == "win32":
        import winsound
        winsound.Beep(2000, 400)  # frequência 1000 Hz, duração 500 ms
    elif sys.platform == "darwin":
        os.system('say "pronto"')
    else:  # Linux e outros
        os.system('echo -e "\a"')

if __name__ == "__main__":
    beep()   # emite som ao terminar com sucesso
