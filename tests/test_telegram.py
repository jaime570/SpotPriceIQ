import pytest
import requests

from src.notificaciones import telegram


def test_avisar_sin_token_devuelve_false(monkeypatch):
    """Sin token no avisa ni falla (el CI no tiene secretos). Se neutraliza load_dotenv:
    si no, en el portátil volvería a leer el .env y enviaría un Telegram de verdad."""
    enviados = []
    monkeypatch.setattr(telegram, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(telegram.requests, "post", lambda *args, **kwargs: enviados.append(kwargs))
    monkeypatch.delenv("TELEGRAM_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

    assert telegram.avisar("hola") is False
    assert enviados == []


class RespuestaFalsa:
    """Imita la respuesta de Telegram cuando todo va bien: raise_for_status() no lanza nada."""
    def raise_for_status(self):
        pass


@pytest.fixture
def telegram_falso(monkeypatch):
    """Telegram configurado (token y chat falsos) pero SIN enviar nada de verdad:
    requests.post se sustituye por una función que apunta el mensaje en `enviados`."""
    enviados = []

    def post_falso(url, data, timeout):
        enviados.append(data)
        return RespuestaFalsa()

    monkeypatch.setattr(telegram, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(telegram.requests, "post", post_falso)
    monkeypatch.setenv("TELEGRAM_TOKEN", "token-falso")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")
    return enviados


def test_avisar_mensaje_corto_se_envia_tal_cual(telegram_falso):
    """Con token, el mensaje llega entero y avisar devuelve True."""
    assert telegram.avisar("✅ Predicción hecha") is True
    assert len(telegram_falso) == 1
    assert telegram_falso[0]["text"] == "✅ Predicción hecha"


def test_avisar_mensaje_largo_se_recorta(telegram_falso):
    """Telegram rechaza mensajes de más de 4096 caracteres (p. ej. un informe de pandera):
    se recorta a 4000 y se avisa de que el detalle está en el log."""
    assert telegram.avisar("x" * 5000) is True
    enviado = telegram_falso[0]["text"]
    assert len(enviado) < 4096
    assert enviado.startswith("x" * 4000)
    assert enviado.endswith("(recortado, detalle completo en el log)")


def test_avisar_fallo_de_red_no_rompe(telegram_falso, monkeypatch):
    """Si Telegram o la red fallan, avisar devuelve False sin lanzar error:
    una alerta nunca debe tumbar el flow que la envía."""
    def post_que_falla(url, data, timeout):
        raise requests.ConnectionError("sin red")

    monkeypatch.setattr(telegram.requests, "post", post_que_falla)
    assert telegram.avisar("hola") is False
