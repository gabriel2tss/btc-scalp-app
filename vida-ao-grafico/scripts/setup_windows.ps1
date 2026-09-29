# Setup no PC Windows (PowerShell), a partir da pasta vida-ao-grafico\
#   powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[mt5,dev]"
.\.venv\Scripts\python -m pytest -q
Write-Host ""
Write-Host "Pronto. Com o MT5 aberto e logado, rode:"
Write-Host "  .\.venv\Scripts\python -m vag.recon"
