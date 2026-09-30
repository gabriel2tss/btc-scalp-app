# Setup no notebook Windows (PowerShell), a partir da pasta vida-ao-grafico\
#   powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
$ErrorActionPreference = "Stop"
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
# PyTorch com CUDA (GTX 1650 = Turing, sm_75). O torch do PyPI no Windows é só CPU.
.\.venv\Scripts\python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python -m pip install --no-cache-dir -e ".[dev]" scikit-learn
.\.venv\Scripts\python -c "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'sem GPU')"
.\.venv\Scripts\python -m pytest -q
