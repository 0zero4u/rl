# engine.py

from collections import deque
from enum import Enum
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from ..config import SETTINGS

class Side(Enum):
    BUY = "BUY"
    SELL = "SELL"

def _side_to_signed_qty(side: str, size: float) -> float:
    """Convert a side string and size to a signed quantity for CVD calculation."""
    s = (side or "").upper()
    if s == Side.BUY.value: return float(size)
    if s == Side.SELL.value: return -float(size)
    return 0.0

class HotCache:
    """
    A stateful calculator that processes tick data to maintain real-time indicators
    across multiple timeframes. It forms the base for all strategy logic.
    """
    def __init__(self):
        self.cfg = SETTINGS
        self.strat_cfg = self.cfg.strategy
        self.current_price: float = np.nan
        self.prev_price: float = np.nan
        self.current_timestamp: Optional[pd.Timestamp] = None
        self.current_funding_rate: float = np.nan
        self.cvd: float = 0.0

        self.bar_data: Dict[str, Dict[str, Any]] = {
            tf: {'open': np.nan, 'high': np.nan, 'low': np.nan, 'close': np.nan,
                 'volume': 0.0, 'cvd': 0.0, 'ts': None, 'pv_sum': 0.0, 'vol_sum': 0.0}
            for tf in self.cfg.TIMEFRAMES
        }

        self.ema_8_15m, self.prev_ema_8_15m = np.nan, np.nan
        self.ema_21_15m, self.prev_ema_21_15m = np.nan, np.nan
        self.ema_50_4h_on_close = np.nan
        self.atr_14_2h, self.atr_14_4h = np.nan, np.nan
        self.prev_close_2h, self.prev_close_4h = np.nan, np.nan

        self.volume_1m_deq = deque(maxlen=60)
        self.cvd_1m_deq = deque(maxlen=60)
        self.prices_15m_deq = deque(maxlen=self.strat_cfg.BB_PERIOD)
        self.bb_widths_15m_deq = deque(maxlen=self.strat_cfg.BBW_HISTORY_PERIOD)
        self.funding_rate_deq = deque(maxlen=self.strat_cfg.FUNDING_RATE_HISTORY_PERIOD)

        self.alphas = {'ema_50_4h': 2 / (50 + 1), 'atr_14': 1 / 14}
        self.default_tick_alphas = {'ema_8_15m': 2 / (8 + 1), 'ema_21_15m': 2 / (21 + 1)}

    @staticmethod
    def _update_ema(val: float, prev: float, alpha: float) -> float:
        return val if pd.isna(prev) else (val * alpha) + (prev * (1 - alpha))

    def _init_bar_if_needed(self, tf: str, price: float):
        bar = self.bar_data[tf]
        if pd.isna(bar['open']):
            bar['open'] = bar['high'] = bar['low'] = price
            bar['ts'] = self.current_timestamp.floor(self.cfg.TIMEFRAMES[tf])

    def _close_and_roll_bar(self, tf: str, close_price: float):
        bar = self.bar_data[tf]
        bar['close'] = close_price
        if tf == '1m':
            self.volume_1m_deq.append(bar['volume'])
            self.cvd_1m_deq.append(bar['cvd'])
        elif tf == '15m':
            self.prices_15m_deq.append(close_price)
            if len(self.prices_15m_deq) == self.strat_cfg.BB_PERIOD:
                prices = np.array(self.prices_15m_deq)
                mean, std = prices.mean(), prices.std()
                if mean > 0: self.bb_widths_15m_deq.append((std * 4) / mean)
            if hasattr(self, '_on_15m_bar_close'): self._on_15m_bar_close(close_price)
        elif tf == '2h':
            if not (pd.isna(bar['high']) or pd.isna(self.prev_close_2h)):
                tr = max(bar['high'] - bar['low'], abs(bar['high'] - self.prev_close_2h), abs(bar['low'] - self.prev_close_2h))
                self.atr_14_2h = self._update_ema(tr, self.atr_14_2h, self.alphas['atr_14'])
            self.prev_close_2h = close_price
        elif tf == '4h':
            self.ema_50_4h_on_close = self._update_ema(close_price, self.ema_50_4h_on_close, self.alphas['ema_50_4h'])
            if not (pd.isna(bar['high']) or pd.isna(self.prev_close_4h)):
                tr = max(bar['high'] - bar['low'], abs(bar['high'] - self.prev_close_4h), abs(bar['low'] - self.prev_close_4h))
                self.atr_14_4h = self._update_ema(tr, self.atr_14_4h, self.alphas['atr_14'])
            self.prev_close_4h = close_price
        
        bar['ts'] += self.cfg.TIMEFRAMES[tf]
        bar['open'] = bar['high'] = bar['low'] = close_price
        bar['volume'] = bar['cvd'] = bar['pv_sum'] = bar['vol_sum'] = 0.0

    def _update_bar(self, tf: str, price: float, size: float, cvd_delta: float):
        self._init_bar_if_needed(tf, price)
        bar = self.bar_data[tf]
        while self.current_timestamp >= bar['ts'] + self.cfg.TIMEFRAMES[tf]:
            self._close_and_roll_bar(tf, self.prev_price if not pd.isna(self.prev_price) else price)
        
        bar['high'] = max(bar['high'], price)
        bar['low'] = min(bar['low'], price)
        bar['volume'] += size
        bar['cvd'] += cvd_delta
        bar['pv_sum'] += price * size
        bar['vol_sum'] += size

    def _calculate_dynamic_alphas(self) -> Dict[str, float]:
        s = self.strat_cfg.lookback
        if not s.ENABLED or pd.isna(self.atr_14_4h) or self.atr_14_4h <= 0:
            return self.default_tick_alphas

        vol_ratio = self.atr_14_4h / s.NORMAL_ATR_4H
        clamped_ratio = np.clip(vol_ratio, s.MIN_VOL_RATIO_CLAMP, s.MAX_VOL_RATIO_CLAMP)
        lookback_8 = np.clip(s.BASE_LOOKBACK_8 / clamped_ratio, s.MIN_LOOKBACK, s.MAX_LOOKBACK)
        lookback_21 = np.clip(s.BASE_LOOKBACK_21 / clamped_ratio, s.MIN_LOOKBACK, s.MAX_LOOKBACK)
        
        return {'ema_8_15m': 2 / (lookback_8 + 1), 'ema_21_15m': 2 / (lookback_21 + 1)}

    def update(self, row: Any):
        self.prev_price, self.current_price = self.current_price, float(row.price)
        self.current_timestamp = pd.Timestamp(row.timestamp)

        if hasattr(row, 'funding_rate') and not pd.isna(row.funding_rate):
            new_fr = float(row.funding_rate)
            if new_fr != self.current_funding_rate:
                self.current_funding_rate = new_fr
                self.funding_rate_deq.append(new_fr)
        
        if not pd.isna(self.prev_price):
            size, side = float(row.size), str(row.side)
            cvd_delta = _side_to_signed_qty(side, size)
            self.cvd += cvd_delta
            
            dynamic_alphas = self._calculate_dynamic_alphas()
            self.prev_ema_8_15m, self.prev_ema_21_15m = self.ema_8_15m, self.ema_21_15m
            self.ema_8_15m = self._update_ema(self.current_price, self.ema_8_15m, dynamic_alphas['ema_8_15m'])
            self.ema_21_15m = self._update_ema(self.current_price, self.ema_21_15m, dynamic_alphas['ema_21_15m'])
            
            for tf in self.cfg.TIMEFRAMES:
                self._update_bar(tf, self.current_price, size, cvd_delta)

    def get_intent_payload(self, side: Side, strategy_id: str) -> Dict[str, Any]:
        regime_z = (self.current_price - self.ema_50_4h_on_close) / self.atr_14_4h \
            if not pd.isna(self.ema_50_4h_on_close) and self.atr_14_4h > 0 else np.nan

        vol_z = cvd_z = np.nan
        if len(self.volume_1m_deq) >= 10:
            vol_mean, vol_std = np.mean(self.volume_1m_deq), np.std(self.volume_1m_deq)
            cvd_mean, cvd_std = np.mean(self.cvd_1m_deq), np.std(self.cvd_1m_deq)
            if vol_std > 1e-9: vol_z = (self.bar_data['1m']['volume'] - vol_mean) / vol_std
            if cvd_std > 1e-9: cvd_z = (self.bar_data['1m']['cvd'] - cvd_mean) / cvd_std
        
        if pd.isna(regime_z): regime_tag = 'UNKNOWN'
        elif regime_z > self.strat_cfg.REGIME_Z_BULL: regime_tag = 'BULL'
        elif regime_z < self.strat_cfg.REGIME_Z_BEAR: regime_tag = 'BEAR'
        else: regime_tag = 'SIDEWAYS'

        event_tier = 'PRIME' if (not pd.isna(vol_z) and vol_z > self.strat_cfg.PRIME_EVENT_Z_SCORE) or \
           (not pd.isna(cvd_z) and abs(cvd_z) > self.strat_cfg.PRIME_EVENT_Z_SCORE) else 'STANDARD'

        funding_z, is_extreme = np.nan, False
        if len(self.funding_rate_deq) >= 10:
            mean_fr, std_fr = np.mean(self.funding_rate_deq), np.std(self.funding_rate_deq)
            if std_fr > 1e-9:
                funding_z = (self.current_funding_rate - mean_fr) / std_fr
                is_extreme = abs(funding_z) > self.strat_cfg.FALLBACK_IGNITION_Z

        return {
            'timestamp': self.current_timestamp, 'asset': self.cfg.ASSET,
            'side': side.value, 'strategy_id': strategy_id,
            'event_tier': event_tier, 'regime_tag': regime_tag,
            'raw_regime_z': regime_z, 'raw_vol_z_1m': vol_z, 'raw_cvd_z_1m': cvd_z,
            'sentiment.funding_rate': self.current_funding_rate,
            'sentiment.funding_z_score': funding_z,
            'sentiment.is_extreme': is_extreme
        }

    def check_trigger(self) -> Optional[Side]:
        raise NotImplementedError("Strategy subclasses must implement check_trigger.")
