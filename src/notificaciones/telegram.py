"""Alertas por Telegram (fallos, puerta horaria, alarma, reentreno, heartbeat de la predicción).

Mismo patrón de secretos que la ingesta (load_dotenv + os.getenv). Dos reglas:
- Opcional: sin TELEGRAM_TOKEN / TELEGRAM_CHAT_ID no hace nada (el CI funciona sin ellos).
- Nunca rompe el pipeline: si Telegram o la red fallan, solo lo deja en el log.
"""
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]


MAX_TELEGRAM = 4000  # Telegram rechaza mensajes > 4096 caracteres; dejamos margen


def avisar(texto: str) -> bool:
    """Envía `texto` a tu chat de Telegram. Devuelve True si se envió.
    Recorta mensajes largos (p. ej. el informe completo de pandera) para que
    Telegram no los rechace: el detalle completo sigue en el log."""
    load_dotenv(ROOT / ".env")
    token = (os.getenv("TELEGRAM_TOKEN") or "").strip().strip("'")
    chat_id = (os.getenv("TELEGRAM_CHAT_ID") or "").strip().strip("'")
    if not token or not chat_id:
        return False  # sin configurar: no avisa, pero tampoco falla

    if len(texto) > MAX_TELEGRAM:
        texto = texto[:MAX_TELEGRAM] + "\n… (recortado, detalle completo en el log)"

    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": texto},
            timeout=10,
        )
        r.raise_for_status()
        return True
    except Exception as e:  # noqa: BLE001 — a propósito: una alerta jamás debe tumbar el flow
        print(f"[telegram] No se pudo enviar el aviso: {e}")
        return False

def aviso_fallo(flow, flow_run, state):
    """Hook de Prefect (on_failure / on_crashed): avisa de cualquier flow que falle."""
    avisar(f"❌ {flow.name} ha fallado\n{state.message}")