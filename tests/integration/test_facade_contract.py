"""Check the small public contract shared by every exchange facade."""

from inspect import signature
from pathlib import Path

import pytest

from veldra.binance.facade import Binance
from veldra.bitget.facade import Bitget
from veldra.bybit.facade import Bybit
from veldra.gate.facade import Gate
from veldra.htx.facade import HTX
from veldra.kucoin.facade import KuCoin
from veldra.okx.facade import OKX
from veldra.upbit.facade import Upbit

type Facade = Binance | Bitget | Bybit | Gate | HTX | KuCoin | OKX | Upbit


@pytest.mark.parametrize(
    "facade_type", [Binance, Bitget, Bybit, Gate, HTX, KuCoin, OKX, Upbit]
)
def test_exchange_facades_share_the_core_retrieval_surface(
    tmp_path: Path, facade_type: type[Facade]
) -> None:
    """Every exchange exposes local configuration and tabular core methods."""
    service = facade_type(tmp_path / facade_type.__name__.lower(), progress=False)
    try:
        assert service.data_dir.is_absolute()
        assert hasattr(service, "earliest_date")
        assert service.max_workers == 32
        for method_name in ("get_klines", "get_trades"):
            parameters = tuple(signature(getattr(service, method_name)).parameters)
            assert parameters[:3] == ("pairs", "start", "end")
    finally:
        close = getattr(service, "close", None)
        if close is not None:
            close()
