from collections import deque
from enum import Enum, auto
from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore
from ..config import SETTINGS

class Side(Enum):
    BUY = "BUY"
    SELL = "SELL"

class MarketState(Enum):
    INITIALIZING = auto()
    RANGE_STABLE = auto()
    RANGE_CONTRACTING = auto()
    TREND_BULL = auto()
    TREND_BEAR = auto()
    CONFIRMED_CHOCH_BULL = auto() # Warning of a new uptrend
    CONFIRMED_CHOCH_BEAR = auto() # Warning of a new downtrend

def _side_to_signed_qty(side: str, size: float) -> float:
    s = (side or "").upper()
    if s == Side.BUY.value: return float(size)
    if s == Side.SELL.value: return -float(size)
    return 0.0

class HotCache:
    """
    A stateful engine that processes tick data to perform market structure analysis,
    detect Smart Money Concepts (SMC), and determine the overall market state.
    """
    def __init__(self):
        self.cfg = SETTINGS
        self.strat_cfg = self.cfg.strategy
        self.smc_cfg = self.strat_cfg.smc
        self.market_state: MarketState = MarketState.INITIALIZING
        
        # --- Core State Variables ---
        self.current_price: float = np.nan
        self.prev_price: float = np.nan
        self.current_timestamp: Optional[pd.Timestamp] = None
        self.current_funding_rate: float = np.nan
        self.cvd: float = 0.0

        # --- Bar & Timeframe Data ---
        self.bar_data: Dict[str, Dict[str, Any]] = {
            tf: {'open': np.nan, 'high': np.nan, 'low': np.nan, 'close': np.nan,
                 'volume': 0.0, 'cvd': 0.0, 'ts': None} for tf in self.cfg.TIMEFRAMES
        }
        self.bars_1h_deq = deque(maxlen=10)
        self.bb_widths_15m_deq = deque(maxlen=self.strat_cfg.BBW_HISTORY_PERIOD)
        self.prices_15m_deq = deque(maxlen=self.strat_cfg.BB_PERIOD)
        self.cvd_1m_deq = deque(maxlen=60) # For order flow Z-score confirmation

        # --- Market Structure (1h) ---
        self.last_swing_high: float = np.nan
        self.last_swing_low: float = np.nan

        # --- SMC Primitives (Active Zones) ---
        self.bullish_fvg_zones = [] # List of {'top': float, 'bottom': float, 'ts': timestamp}
        self.bearish_fvg_zones = []
        self.bullish_ob_zones = []
        self.bearish_ob_zones = []

        # --- VWAP & Volume ---
        self.daily_pv_sum: float = 0.0
        self.daily_vol_sum: float = 0.0
        self.last_known_day: Optional[int] = None
        self.volume_ema_1h: Optional[float] = None
        self.atr_4h: float = np.nan # Use a simple ATR for feature normalization
        self.prev_close_4h: float = np.nan
        
    def _init_bar_if_needed(self, tf: str, price: float):
        bar = self.bar_data[tf]
        if pd.isna(bar['open']):
            bar['open'] = bar['high'] = bar['low'] = price
            bar['ts'] = self.current_timestamp.floor(self.cfg.TIMEFRAMES[tf])

    def _update_daily_vwap(self, price: float, size: float):
        current_day = self.current_timestamp.dayofyear
        if self.last_known_day != current_day:
            self.daily_pv_sum, self.daily_vol_sum = 0.0, 0.0
            self.last_known_day = current_day
        self.daily_pv_sum += price * size
        self.daily_vol_sum += size

    def get_daily_vwap(self) -> float:
        return (self.daily_pv_sum / self.daily_vol_sum) if self.daily_vol_sum > 1e-9 else np.nan

    def _analyze_closed_bar_for_smc_signals(self):
        """Streaming equivalent of the vectorized SMC signal detection function."""
        if len(self.bars_1h_deq) < 4: return

        b0, b1, b2 = self.bars_1h_deq[-1], self.bars_1h_deq[-2], self.bars_1h_deq[-3]

        # --- 1. Pivot Point Detection (Structure) ---
        pivot_range = self.smc_cfg.PIVOT_LOOKUP * 2 + 1
        if len(self.bars_1h_deq) >= pivot_range:
            window = list(self.bars_1h_deq)[-pivot_range:]
            middle_bar = window[self.smc_cfg.PIVOT_LOOKUP]
            if max(b['high'] for b in window) == middle_bar['high']: self.last_swing_high = middle_bar['high']
            if min(b['low'] for b in window) == middle_bar['low']: self.last_swing_low = middle_bar['low']

        # --- 2. BoS / CHOCH Signal Detection ---
        bullish_bos, bearish_bos = False, False
        if not pd.isna(self.last_swing_high) and b1['close'] < self.last_swing_high and b0['close'] >= self.last_swing_high:
            bullish_bos = True
        if not pd.isna(self.last_swing_low) and b1['close'] > self.last_swing_low and b0['close'] <= self.last_swing_low:
            bearish_bos = True

        # --- 3. FVG (Fair Value Gap) Detection ---
        if b0['low'] > b2['high']: self.bullish_fvg_zones.append({'top': b0['low'], 'bottom': b2['high'], 'ts': b0['ts']})
        if b0['high'] < b2['low']: self.bearish_fvg_zones.append({'top': b2['low'], 'bottom': b0['high'], 'ts': b0['ts']})

        # --- 4. OB (Order Block) Detection ---
        if (b1['close'] < b1['open']) and (b0['close'] > b0['open']) and b0['close'] > b1['high']:
            self.bullish_ob_zones.append({'top': b1['high'], 'bottom': b1['low'], 'ts': b1['ts']})
        if (b1['close'] > b1['open']) and (b0['close'] < b0['open']) and b0['close'] < b1['low']:
            self.bearish_ob_zones.append({'top': b1['high'], 'bottom': b1['low'], 'ts': b1['ts']})
        
        self._update_market_state(bullish_bos, bearish_bos)
        self._prune_mitigated_zones(b0['close'])

    def _update_market_state(self, bullish_bos: bool, bearish_bos: bool):
        """The 'General' - uses tactical signals to set the strategic bias."""
        if self.market_state == MarketState.TREND_BULL and bearish_bos: self.market_state = MarketState.CONFIRMED_CHOCH_BEAR
        elif self.market_state == MarketState.TREND_BEAR and bullish_bos: self.market_state = MarketState.CONFIRMED_CHOCH_BULL
        elif bullish_bos: self.market_state = MarketState.TREND_BULL
        elif bearish_bos: self.market_state = MarketState.TREND_BEAR
        else: # Logic for ranging states
            if len(self.bb_widths_15m_deq) == self.bb_widths_15m_deq.maxlen:
                pct_rank = percentileofscore(list(self.bb_widths_15m_deq), self.bb_widths_15m_deq[-1])
                if pct_rank < self.strat_cfg.BBW_PCT_RANK_CONTRACT:
                    self.market_state = MarketState.RANGE_CONTRACTING
                else:
                    self.market_state = MarketState.RANGE_STABLE

    def _prune_mitigated_zones(self, price: float):
        """Removes SMC zones that price has already traded through."""
        self.bullish_fvg_zones = [z for z in self.bullish_fvg_zones if price > z['bottom']]
        self.bearish_fvg_zones = [z for z in self.bearish_fvg_zones if price < z['top']]
        self.bullish_ob_zones = [z for z in self.bullish_ob_zones if price > z['bottom']]
        self.bearish_ob_zones = [z for z in self.bearish_ob_zones if price < z['top']]

    def _close_and_roll_bar(self, tf: str, close_price: float):
        bar = self.bar_data[tf]
        bar['close'] = close_price
        
        if tf == '1m': self.cvd_1m_deq.append(bar['cvd'])
        elif tf == '15m':
            self.prices_15m_deq.append(close_price)
            if len(self.prices_15m_deq) == self.strat_cfg.BB_PERIOD:
                prices = np.array(self.prices_15m_deq)
                mean, std = prices.mean(), prices.std()
                if mean > 0: self.bb_widths_15m_deq.append((std * 4) / mean)
        elif tf == self.smc_cfg.STRUCTURE_TIMEFRAME: # '1h'
            self.bars_1h_deq.append(bar.copy())
            self._analyze_closed_bar_for_smc_signals()
        elif tf == '4h':
            if not (pd.isna(bar['high']) or pd.isna(self.prev_close_4h)):
                tr = max(bar['high'] - bar['low'], abs(bar['high'] - self.prev_close_4h), abs(bar['low'] - self.prev_close_4h))
                self.atr_4h = tr if pd.isna(self.atr_4h) else (self.atr_4h * 13 + tr) / 14
            self.prev_close_4h = close_price
        
        bar['ts'] += self.cfg.TIMEFRAMES[tf]
        bar['open'] = bar['high'] = bar['low'] = close_price
        bar['volume'] = bar['cvd'] = 0.0

    def _update_bar(self, tf: str, price: float, size: float, cvd_delta: float):
        self._init_bar_if_needed(tf, price)
        bar = self.bar_data[tf]
        while self.current_timestamp >= bar['ts'] + self.cfg.TIMEFRAMES[tf]:
            self._close_and_roll_bar(tf, self.prev_price if not pd.isna(self.prev_price) else price)
        
        bar['high'] = max(bar['high'], price); bar['low'] = min(bar['low'], price)
        bar['volume'] += size; bar['cvd'] += cvd_delta

    def update(self, row: Any):
        self.prev_price, self.current_price = self.current_price, float(row.price)
        self.current_timestamp = pd.Timestamp(row.timestamp)
        if hasattr(row, 'funding_rate') and not pd.isna(row.funding_rate): self.current_funding_rate = float(row.funding_rate)

        if not pd.isna(self.prev_price):
            size, side = float(row.size), str(row.side)
            self._update_daily_vwap(self.current_price, size)
            cvd_delta = _side_to_signed_qty(side, size)
            self.cvd += cvd_delta
            for tf in self.cfg.TIMEFRAMES: self._update_bar(tf, self.current_price, size, cvd_delta)

    def get_intent_payload(self, side: Side, strategy_id: str) -> Dict[str, Any]:
        """Compiles all calculated features into a dictionary for the model."""
        vwap = self.get_daily_vwap()
        cvd_z = np.nan
        if len(self.cvd_1m_deq) > 10:
            mean, std = np.mean(self.cvd_1m_deq), np.std(self.cvd_1m_deq)
            if std > 1e-9: cvd_z = (self.bar_data['1m']['cvd'] - mean) / std

        bbw_pct = np.nan
        if len(self.bb_widths_15m_deq) > 50:
             bbw_pct = percentileofscore(list(self.bb_widths_15m_deq), self.bb_widths_15m_deq[-1])

        # Calculate normalized distances to nearest zones
        dist_fvg, dist_ob = np.nan, np.nan
        norm = self.atr_4h if not pd.isna(self.atr_4h) and self.atr_4h > 0 else self.current_price * 0.01
        
        if side == Side.BUY:
            if self.bullish_fvg_zones: dist_fvg = (self.current_price - self.bullish_fvg_zones[-1]['top']) / norm
            if self.bullish_ob_zones: dist_ob = (self.current_price - self.bullish_ob_zones[-1]['top']) / norm
        else: # SELL
            if self.bearish_fvg_zones: dist_fvg = (self.bearish_fvg_zones[-1]['bottom'] - self.current_price) / norm
            if self.bearish_ob_zones: dist_ob = (self.bearish_ob_zones[-1]['bottom'] - self.current_price) / norm

        return {
            'timestamp': self.current_timestamp, 'asset': self.cfg.ASSET,
            'side': side.value, 'strategy_id': strategy_id,
            'market_state': self.market_state.name,
            'price_to_vwap_ratio': (self.current_price / vwap) if vwap > 0 else np.nan,
            'bbw_15m_percentile': bbw_pct,
            'raw_cvd_z_1m': cvd_z,
            'distance_to_fvg_norm': dist_fvg,
            'distance_to_ob_norm': dist_ob,
            'sentiment.funding_z_score': np.nan, # Placeholder, funding logic can be added back
        }

    def check_trigger(self) -> Optional[Side]:
        raise NotImplementedError("Strategy subclasses must implement check_trigger.")
