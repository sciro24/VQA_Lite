import sys
import subprocess
from pathlib import Path
import torch
from common.config import cfg, DEVICE, DATA_DIR
from common.utils import set_seed


def main():
    required_pkgs = [
        'sentence-transformers', 'tqdm', 'PyYAML', 'grad-cam',
        'opencv-python-headless', 'matplotlib'
    ]
    if '--force-install' in sys.argv:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q'] + required_pkgs)

    print('\n--- Informazioni Ambiente ---')
    print('Python:', sys.version.split()[0])
    print('Torch:', torch.__version__)
    print('CUDA available:', torch.cuda.is_available())

    set_seed(cfg['training']['seed'])
    Path(DATA_DIR).mkdir(exist_ok=True)
    print(f"Dispositivo selezionato: {DEVICE}")
    print(f"Path salvataggio modello: {cfg['paths']['model_save_path']}")

    # auto-run next
    import section2_utils as next_section
    next_section.main()


if __name__ == '__main__':
    main()
