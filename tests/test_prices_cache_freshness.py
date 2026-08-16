"""Politica de frescura del cache de precios (Bloque 2, 2026-08-16).

Un cache mas viejo que _CACHE_MAX_AGE_DAYS se ignora y se re-descarga, para no
publicar precios rancios (bug: una corrida uso un cache de ~5 semanas). El test
no toca la red: verifica solo el helper _cache_is_fresh con mtimes controlados.
"""

from __future__ import annotations

import os
import time

from irfn.data.prices import _CACHE_MAX_AGE_DAYS, _cache_is_fresh


def test_cache_ausente_no_es_fresco(tmp_path):
    assert _cache_is_fresh(tmp_path / "no_existe.parquet") is False


def test_cache_reciente_es_fresco(tmp_path):
    p = tmp_path / "close_SPY_2013-01-01.parquet"
    p.write_bytes(b"x")
    assert _cache_is_fresh(p) is True


def test_cache_viejo_no_es_fresco(tmp_path):
    p = tmp_path / "close_SPY_2013-01-01.parquet"
    p.write_bytes(b"x")
    # backdatear el mtime a (_CACHE_MAX_AGE_DAYS + 1) dias atras
    old = time.time() - (_CACHE_MAX_AGE_DAYS + 1) * 86400.0
    os.utime(p, (old, old))
    assert _cache_is_fresh(p) is False


def test_borde_justo_dentro_del_umbral(tmp_path):
    p = tmp_path / "close_SPY_2013-01-01.parquet"
    p.write_bytes(b"x")
    # 1 hora antes del umbral: sigue fresco
    almost = time.time() - (_CACHE_MAX_AGE_DAYS * 86400.0 - 3600.0)
    os.utime(p, (almost, almost))
    assert _cache_is_fresh(p) is True
