"""Exercise project-relative cache paths across public exchange facades."""

from pathlib import Path

import pytest

from veldra import Binance, Bitget, Bybit, Gate, HTX, KuCoin, OKX, Upbit

type Facade = Binance | Bitget | Bybit | Gate | HTX | KuCoin | OKX | Upbit


@pytest.mark.parametrize(
    "facade", (Binance, Bitget, Bybit, Gate, HTX, KuCoin, OKX, Upbit)
)
def test_default_cache_belongs_to_calling_project(
    facade: type[Facade], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resolve the default and an explicit legacy path from the calling project."""
    project = tmp_path / "myproject"
    project.mkdir()
    monkeypatch.chdir(project)

    default = facade()
    legacy = facade(data_dir="data")
    assert default.data_dir == project / "data" / "veldra"
    assert legacy.data_dir == project / "data"
    assert not (project / "data").exists()

    monkeypatch.chdir(tmp_path)
    assert default.data_dir == project / "data" / "veldra"
    assert facade().data_dir == tmp_path / "data" / "veldra"
