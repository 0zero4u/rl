

import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List
import pandas as pd

# --- TRADING STRATEGY CONFIGURATION ---

@dataclass(frozen=True)
class LookbackConfig:
    ENABLED: bool = True
    NORMAL_ATR_4H: float = 250.0
    BASE_LOOKBACK_8: int = 30
    BASE_LOOKBACK_21: int = 60
    MIN_LOOKBACK: int = 10
    MAX_LOOKBACK: int = 120
    MIN_VOL_RATIO_CLAMP: float = 0.5
    MAX_VOL_RATIO_CLAMP: float = 2.0

@dataclass(frozen=True)
class IgnitionConfig:
    ENABLED: bool = True
    BASE_Z: float = 2.0
    MIN_Z: float = 1.75
    MAX_Z: float = 4.0
    ATR_SENSITIVITY: float = 0.5

@dataclass(frozen=True)
class TrendConfig:
    ENABLED: bool = True
    BASE_BPS: float = 3.0
    ATR_SENSITIVITY: float = 50.0

@dataclass(frozen=True)
class TradingStrategyConfig:
    lookback: LookbackConfig = field(default_factory=LookbackConfig)
    ignition: IgnitionConfig = field(default_factory=IgnitionConfig)
    trend: TrendConfig = field(default_factory=TrendConfig)
    FALLBACK_IGNITION_Z: float = 2.5
    FALLBACK_TREND_BPS_PER_BAR: float = 5.0
    BBW_PCT_RANK_CONTRACT: float = 10.0
    ATR_2H_TO_BAND_MULT: float = 1.5
    BB_PERIOD: int = 20
    BBW_HISTORY_PERIOD: int = 250
    TREND_SLOPE_PERIOD: int = 5
    FUNDING_RATE_HISTORY_PERIOD: int = 24
    REGIME_Z_BULL: float = 1.0
    REGIME_Z_BEAR: float = -1.0
    PRIME_EVENT_Z_SCORE: float = 3.0

# --- LABELING CONFIGURATION ---

@dataclass(frozen=True)
class LabelingConfig:
    INTENTS_FILE_NAME: str = "all_trade_intents_for_training.parquet"
    OUTPUT_FILE_NAME: str = "labeled_trade_intents.parquet"
    BAR_TIMEFRAME: str = "1T"
    ATR_LOOKBACK: int = 20
    PROFIT_TAKE_ATR: float = 2.0
    STOP_LOSS_ATR: float = 2.0
    TIME_LIMIT_MINUTES: int = 60

# --- MODEL TRAINING CONFIGURATION ---

@dataclass(frozen=True)
class TrainingConfig:
    MODEL_OUTPUT_FILE: str = "lgbm_classifier_model.joblib"
    FEATURE_COLUMNS: List[str] = field(default_factory=lambda: [
        'raw_regime_z', 'raw_vol_z_1m', 'raw_cvd_z_1m',
        'sentiment.funding_rate', 'sentiment.funding_z_score',
        'sentiment.is_extreme', 'strategy_id'
    ])
    TARGET_COLUMN: str = 'label'
    TEST_SET_SIZE: float = 0.20
    LGBM_PARAMS: Dict = field(default_factory=lambda: {
        'objective': 'multiclass', 'num_class': 3, 'metric': 'multi_logloss',
        'n_estimators': 1000, 'learning_rate': 0.05, 'num_leaves': 31,
        'max_depth': -1, 'seed': 42, 'n_jobs': -1, 'verbose': -1,
        'colsample_bytree': 0.8, 'subsample': 0.8,
    })

# --- GLOBAL SYSTEM & DATA CONFIGURATION ---

@dataclass(frozen=True)
class GlobalConfig:
    # --- Paths & Asset ---
    BASE_PATH: str = os.getenv('BASE_PATH', "/content/drive/MyDrive/crypto_data/alpha_proof")
    ASSET: str = "BTCUSDT"

    # --- Time Periods ---
    IN_SAMPLE_START: datetime = datetime(2025, 1, 1)
    IN_SAMPLE_END: datetime = datetime(2025, 5, 31)
    OUT_OF_SAMPLE_START: datetime = datetime(2025, 6, 1)
    OUT_OF_SAMPLE_END: datetime = datetime(2025, 7, 31)

    # --- Data Schemas & Processing ---
    BINANCE_RAW_COLUMNS: List[str] = field(default_factory=lambda: ['id', 'price', 'qty', 'quote_qty', 'time', 'is_buyer_maker'])
    DTYPE_MAP: Dict[str, str] = field(default_factory=lambda: {
        'id': 'int64', 'price': 'float64', 'qty': 'float64',
        'quote_qty': 'float64', 'time': 'int64', 'is_buyer_maker': 'bool'
    })
    FINAL_COLUMNS: List[str] = field(default_factory=lambda: ['timestamp', 'asset', 'trade_id', 'price', 'size', 'side'])
    BATCH_SIZE: int = 8_000_000
    PARQUET_COMPRESSION: str = "zstd"
    TIMEFRAMES: Dict[str, pd.Timedelta] = field(default_factory=lambda: {
        '1m': pd.Timedelta(minutes=1), '3m': pd.Timedelta(minutes=3),
        '15m': pd.Timedelta(minutes=15), '1h': pd.Timedelta(hours=1),
        '2h': pd.Timedelta(hours=2), '4h': pd.Timedelta(hours=4),
    })

    # --- Sub-configurations ---
    strategy: TradingStrategyConfig = field(default_factory=TradingStrategyConfig)
    labeling: LabelingConfig = field(default_factory=LabelingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # --- Directory Methods ---
    def _get_period_path(self, period_name: str) -> str:
        return os.path.join(self.BASE_PATH, period_name)

    def get_raw_trades_path(self, period_name: str) -> str:
        return os.path.join(self._get_period_path(period_name), "raw", "trades")

    def get_processed_trades_path(self, period_name: str) -> str:
        return os.path.join(self._get_period_path(period_name), "processed", "trades")

    def get_funding_rate_path(self) -> str:
        return os.path.join(self.BASE_PATH, "fundingrate")

    def get_intents_path(self, period_name: str, strategy_name: str) -> str:
        filename = f"trade_intents_{period_name}_{strategy_name}.parquet"
        return os.path.join(self._get_period_path(period_name), filename)

    def get_combined_intents_path(self) -> str:
        return os.path.join(self.BASE_PATH, self.labeling.INTENTS_FILE_NAME)

    def get_labeled_data_path(self) -> str:
        return os.path.join(self.BASE_PATH, self.labeling.OUTPUT_FILE_NAME)

    def get_model_path(self) -> str:
        return os.path.join(self.BASE_PATH, self.training.MODEL_OUTPUT_FILE)

# --- Singleton Instance ---
SETTINGS = GlobalConfig()
