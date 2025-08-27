# strategies.py

from typing import Optional
import numpy as np
import pandas as pd
from scipy.stats import linregress, percentileofscore
from collections import deque
from .engine import HotCache, Side

class PureMeanReversion(HotCache):
    def check_trigger(self) -> Optional[Side]:
        vwap_3m = self.get_vwap('3m')
        if pd.isna(self.atr_14_2h) or pd.isna(vwap_3m):
            return None
        
        atr_band = self.strat_cfg.ATR_2H_TO_BAND_MULT * self.atr_14_2h
        lower, upper = vwap_3m - atr_band, vwap_3m + atr_band
        
        if self.prev_price >= lower and self.current_price < lower: return Side.BUY
        if self.prev_price <= upper and self.current_price > upper: return Side.SELL
        return None

class PureMomentumChaser(HotCache):
    def _get_dynamic_ignition_z(self) -> float:
        s = self.strat_cfg.ignition
        if not s.ENABLED or pd.isna(self.atr_14_4h):
            return self.strat_cfg.FALLBACK_IGNITION_Z
        
        vol_ratio = self.atr_14_4h / self.strat_cfg.lookback.NORMAL_ATR_4H
        dynamic_z = s.BASE_Z + ((vol_ratio - 1.0) * s.ATR_SENSITIVITY)
        return np.clip(dynamic_z, s.MIN_Z, s.MAX_Z)

    def check_trigger(self) -> Optional[Side]:
        if len(self.volume_1m_deq) < 30: return None
        
        vwap_3m = self.get_vwap('3m')
        if pd.isna(vwap_3m): return None
        
        vol_mean, vol_std = np.mean(self.volume_1m_deq), np.std(self.volume_1m_deq)
        cvd_mean, cvd_std = np.mean(self.cvd_1m_deq), np.std(self.cvd_1m_deq)
        if vol_std <= 1e-9 or cvd_std <= 1e-9: return None
        
        vol_z = (self.bar_data['1m']['volume'] - vol_mean) / vol_std
        cvd_z = (self.bar_data['1m']['cvd'] - cvd_mean) / cvd_std

        ignition_threshold = self._get_dynamic_ignition_z()
        if not ((vol_z > ignition_threshold) and (abs(cvd_z) > ignition_threshold)): return None
        
        if self.current_price > vwap_3m and cvd_z > 0: return Side.BUY
        if self.current_price < vwap_3m and cvd_z < 0: return Side.SELL
        return None

class ContextAwareCache(HotCache):
    def __init__(self):
        super().__init__()
        self.ema_21_15m_on_close = np.nan
        self.ema_21_15m_deq = deque(maxlen=self.strat_cfg.TREND_SLOPE_PERIOD)
        self.is_trending = False
        self.is_vol_contracting = False
        
    def _get_dynamic_trend_threshold(self) -> float:
        s = self.strat_cfg.trend
        if not s.ENABLED or pd.isna(self.atr_14_4h) or self.current_price <= 0:
            return self.strat_cfg.FALLBACK_TREND_BPS_PER_BAR
            
        atr_as_bps = (self.atr_14_4h / self.current_price) * 10_000
        return max(s.BASE_BPS, s.BASE_BPS + (atr_as_bps * s.ATR_SENSITIVITY))

    def _on_15m_bar_close(self, close_price: float):
        self.ema_21_15m_on_close = self._update_ema(close_price, self.ema_21_15m_on_close, self.default_tick_alphas['ema_21_15m'])
        if not pd.isna(self.ema_21_15m_on_close): self.ema_21_15m_deq.append(self.ema_21_15m_on_close)
        
        if len(self.ema_21_15m_deq) == self.ema_21_15m_deq.maxlen:
            y = list(self.ema_21_15m_deq)
            slope, _, r_val, _, _ = linregress(range(len(y)), y)
            mean_ema = np.mean(y)
            if mean_ema > 0 and r_val**2 > 0.5:
                norm_slope_bps = (slope / mean_ema) * 10_000.0
                self.is_trending = abs(norm_slope_bps) > self._get_dynamic_trend_threshold()
            else:
                self.is_trending = False
                
        if len(self.bb_widths_15m_deq) > 50:
            pct_rank = percentileofscore(list(self.bb_widths_15m_deq), self.bb_widths_15m_deq[-1], kind='strict')
            self.is_vol_contracting = pct_rank < self.strat_cfg.BBW_PCT_RANK_CONTRACT
        else:
            self.is_vol_contracting = False

class ContextAwareMeanReversion(ContextAwareCache, PureMeanReversion):
    def check_trigger(self) -> Optional[Side]:
        if self.is_trending: return None
        return super().check_trigger()

class ContextAwareMomentum(ContextAwareCache, PureMomentumChaser):
    def check_trigger(self) -> Optional[Side]:
        if not self.is_trending and not self.is_vol_contracting: return None
        return super().check_trigger()
