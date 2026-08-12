"""Защита от SSRF при скачивании файлов по URL, пришедшему из запроса.

Зачем: /extract-specs-by-url принимает file_url извне и сервис сам идёт по
нему GET-запросом. Без проверок это позволяло запросить метаданные облака
(169.254.169.254 у Yandex Cloud/AWS — там лежат IAM-токены сервисного
аккаунта) и сканировать внутреннюю сеть compose (redis:6379, postgres:5432,
api-gateway:8000 с его callback-эндпоинтами).

Проверяется РЕЗОЛВНУТЫЙ IP, а не строка хоста: домен под контролем
атакующего может иметь A-запись на 169.254.169.254, и проверка по имени
такое пропустила бы. Редиректы вызывающая сторона обязана отключить —
иначе разрешённый хост редиректит на запрещённый уже после проверки
(этот модуль сам по себе от того случая не спасает).
"""

from __future__ import annotations

import ipaddress
import os
import socket
from urllib.parse import urlparse

ALLOWED_SCHEMES = {"http", "https"}

# Штатный источник файлов — presigned URL объектного хранилища, и во внутренней
# сети compose это http://minio:9000, то есть ПРИВАТНЫЙ адрес. Без явного
# allowlist проверка ниже сломала бы обычную работу сервиса, поэтому хосты
# хранилища разрешены поимённо. Значение берётся из того же S3_ENDPOINT, что
# используется для генерации ссылок; несколько хостов — через запятую.
_DEFAULT_ALLOWED_HOSTS = "minio,storage.yandexcloud.net"


def _allowed_hosts() -> set[str]:
    raw = os.environ.get("FILE_URL_ALLOWED_HOSTS", _DEFAULT_ALLOWED_HOSTS)
    hosts = {h.strip().lower() for h in raw.split(",") if h.strip()}
    endpoint = os.environ.get("S3_ENDPOINT", "")
    if endpoint:
        endpoint_host = urlparse(endpoint).hostname
        if endpoint_host:
            hosts.add(endpoint_host.lower())
    return hosts


class UrlNotAllowed(ValueError):
    """URL указывает на непубличный адрес либо имеет неподдерживаемую схему."""


def _is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    # is_private покрывает 10/8, 172.16/12, 192.168/16, fc00::/7;
    # link_local — 169.254/16 и fe80::/10 (метаданные облака);
    # loopback — 127/8 и ::1; плюс явно неприсваиваемые диапазоны.
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _resolve_all(host: str, port: int) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise UrlNotAllowed(f"Не удалось разрезолвить хост {host!r}: {exc}") from exc
    return [info[4][0] for info in infos]


def validate_public_url(url: str) -> None:
    """Бросает UrlNotAllowed, если url не http(s) или ведёт на непубличный IP.

    Проверяются ВСЕ адреса, в которые резолвится хост: если хотя бы один
    непубличный, запрос отклоняется — иначе DNS с несколькими A-записями
    (публичная + 169.254.169.254) позволял бы попасть на приватный адрес
    в зависимости от того, какой адрес выберет httpx при подключении.
    """
    parsed = urlparse(url)
    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise UrlNotAllowed(f"Схема {parsed.scheme!r} запрещена, разрешены только http/https")

    host = parsed.hostname
    if not host:
        raise UrlNotAllowed("URL без хоста")

    # Хост объектного хранилища разрешён явно: он приватный по своей природе
    # (minio во внутренней сети), и без этого исключения штатная работа
    # сломалась бы. Всё остальное проходит проверку по IP ниже.
    if host.lower() in _allowed_hosts():
        return

    port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)

    # Хост может быть литеральным IP — тогда getaddrinfo вернёт его же,
    # отдельная ветка не нужна, но проверку делаем до резолва, чтобы не
    # ходить в DNS ради заведомо запрещённого литерала.
    try:
        literal_ip = ipaddress.ip_address(host)
    except ValueError:
        literal_ip = None
    if literal_ip is not None and _is_blocked_ip(literal_ip):
        raise UrlNotAllowed(f"Адрес {host} принадлежит непубличному диапазону")

    for address in _resolve_all(host, port):
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            continue
        if _is_blocked_ip(ip):
            raise UrlNotAllowed(
                f"Хост {host!r} резолвится в непубличный адрес {address}"
            )
