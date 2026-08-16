"""Asegura que `import irfn` funcione sin depender de que el paquete este
instalado (pip install -e .). El README recomienda instalar en modo
editable; esto es una red de seguridad para correr pytest directo.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
