"""标的代码标准化测试。"""
import pytest

from qtrade.data.schema import normalize_symbol, split_symbol


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("600519", "600519.SH"),
        ("600519.SH", "600519.SH"),
        ("sh.600519", "600519.SH"),
        ("SH600519", "600519.SH"),
        ("000001", "000001.SZ"),
        ("000001.SZ", "000001.SZ"),
        ("300750", "300750.SZ"),
        ("688981", "688981.SH"),
    ],
)
def test_normalize_ok(raw, expected):
    assert normalize_symbol(raw) == expected


def test_normalize_invalid():
    with pytest.raises(ValueError):
        normalize_symbol("ABCDEF")


def test_split():
    assert split_symbol("600519.SH") == ("600519", "SH")
