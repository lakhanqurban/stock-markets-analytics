"""Central configuration for the hourly crypto algo-trading project.

All paths, the asset universe, time windows and simulation settings live here
so that every pipeline stage (download -> transform -> train -> simulate)
reads exactly the same constants.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"              # per-symbol raw API dumps (parquet)
OUTPUT_DIR = PROJECT_DIR / "outputs"    # predictions / trade lists / reports
for _d in (RAW_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Data source: Binance USDT-M futures (public REST, no API key)
# --------------------------------------------------------------------------- #
BASE_FAPI = "https://fapi.binance.com"
INTERVAL = "1h"
START = "2021-01-01T00:00:00Z"          # window start (UTC)
REQUEST_SLEEP = 0.12                    # seconds between API calls (rate limit)
MAX_RETRIES = 5

# Curated *pure-crypto* USDT perpetuals. Candidates that are not listed on
# Binance futures any more (e.g. delisted MATIC) are filtered out at download
# time against /fapi/v1/exchangeInfo. Tokenized-stock / metal perps that Binance
# now lists (NVDAUSDT, XAUUSDT, SOXLUSDT, ...) are deliberately excluded.
UNIVERSE_CANDIDATES = [
    # large caps
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
    "ADAUSDT", "TRXUSDT", "LTCUSDT", "BCHUSDT", "LINKUSDT", "DOTUSDT",
    # mid caps - L1 / L2 ecosystems
    "AVAXUSDT", "NEARUSDT", "ATOMUSDT", "APTUSDT", "SUIUSDT", "INJUSDT",
    "SEIUSDT", "TONUSDT", "ICPUSDT", "ALGOUSDT", "XTZUSDT", "EOSUSDT",
    "FILUSDT", "HBARUSDT", "XLMUSDT", "VETUSDT", "GRTUSDT", "STXUSDT",
    "ARBUSDT", "OPUSDT", "TIAUSDT", "IMXUSDT",
    # DeFi / app coins
    "UNIUSDT", "AAVEUSDT", "LDOUSDT", "CRVUSDT", "MKRUSDT", "RUNEUSDT",
    "ENAUSDT", "JUPUSDT",
    # memes / high-beta retail
    "1000PEPEUSDT", "1000SHIBUSDT", "WIFUSDT",
]

# --------------------------------------------------------------------------- #
# Target / trading horizon
# --------------------------------------------------------------------------- #
HOLD_HOURS = 24            # forward horizon for the prediction target
TARGET_THRESHOLD = 1.0     # target_24h = growth_future_24h > threshold
TARGET_CLASS_SHARE = 0.5   # expected ~50% positive class with threshold 1.0
FEE_BPS = 10               # one-way fee in basis points (taker)
SLIPPAGE_BPS = 5           # one-way slippage assumption
START_CAPITAL = 10_000.0   # starting capital for the exact (iterative) sim

# --------------------------------------------------------------------------- #
# Time splits (purely chronological - no shuffle)
#   train : 2021-01-01 .. 2024-06-30
#   valid : 2024-07-01 .. 2025-06-30
#   test  : 2025-07-01 .. latest available
# --------------------------------------------------------------------------- #
TRAIN_END = "2024-07-01"
VALID_END = "2025-07-01"

# --------------------------------------------------------------------------- #
# Simulation settings
# --------------------------------------------------------------------------- #
MAX_CONCURRENT_POSITIONS = 5     # sim4: cap on simultaneous longs
STOP_LOSS_PCT = 0.05             # sim4: -5% from entry -> force exit
VOL_TERCILE_MIN = 0.0            # sim4: regime filter, trade only when
#                                 #       BTC realized vol (24h) top-2 terciles
