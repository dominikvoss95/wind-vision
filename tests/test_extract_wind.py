"""Unit tests for OCR parsing logic."""

from wind_vision.data.extract_wind import parse_ocr_text


def test_standard_values():
    assert parse_ocr_text("Wind: 8 kts Böen: 14 kts") == (8, 14)


def test_missing_colon():
    assert parse_ocr_text("Wind 12 kts Böen 20 kts") == (12, 20)


def test_umlaut_fallback():
    assert parse_ocr_text("Wind: 5 kts Boen: 9 kts") == (5, 9)


def test_wind_only():
    wind, gust = parse_ocr_text("Wind: 7 kts")
    assert wind == 7
    assert gust is None


def test_gust_only():
    wind, gust = parse_ocr_text("Böen: 15 kts")
    assert wind is None
    assert gust == 15


def test_no_match():
    assert parse_ocr_text("Temperatur: 22°C") == (None, None)


def test_zero_values():
    assert parse_ocr_text("Wind: 0 kts Böen: 0 kts") == (0, 0)
