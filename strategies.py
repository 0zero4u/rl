from typing import Optional
import numpy as np
import pandas as pd
from .engine import HotCache, Side, MarketState

class SMC_Momentum_Engine(HotCache):
    """
    Looks for trend-continuation trades (BoS).
    This version implements the "Conservative Confirmation" protocol:
    entering on a pullback to a high-probability FVG created during the trend.
    """
    def check_trigger(self) -> Optional[Side]:
        # --- Strategic Filter: Only operate in a confirmed trend ---
        if self.market_state not in [MarketState.TREND_BULL, MarketState.TREND_BEAR]:
            return None

        # Calculate order flow Z-score for confirmation
        cvd_z = np.nan
        if len(self.cvd_1m_deq) > 10:
            mean, std = np.mean(self.cvd_1m_deq), np.std(self.cvd_1m_deq)
            if std > 1e-9: cvd_z = (self.bar_data['1m']['cvd'] - mean) / std
        if pd.isna(cvd_z): return None

        # --- Tactical Entry: Pullback to FVG ---
        if self.market_state == MarketState.TREND_BULL and self.bullish_fvg_zones:
            target_zone = self.bullish_fvg_zones[-1] # Target the most recent FVG
            # Check if price has just entered the zone from above
            if self.prev_price > target_zone['top'] and self.current_price <= target_zone['top']:
                # Final confirmation: Is order flow supporting the move?
                if cvd_z > self.smc_cfg.CVD_Z_SCORE_THRESHOLD:
                    return Side.BUY

        if self.market_state == MarketState.TREND_BEAR and self.bearish_fvg_zones:
            target_zone = self.bearish_fvg_zones[-1]
            if self.prev_price < target_zone['bottom'] and self.current_price >= target_zone['bottom']:
                if cvd_z < -self.smc_cfg.CVD_Z_SCORE_THRESHOLD:
                    return Side.SELL
                    
        return None

class SMC_Reversal_Sniper(HotCache):
    """
    Waits for a trend to show weakness (CHOCH) and targets the first pullback
    to a high-probability reversal zone (the OB/FVG created by the CHOCH).
    """
    def check_trigger(self) -> Optional[Side]:
        # --- Strategic Filter: Only operate after a confirmed CHOCH ---
        if self.market_state not in [MarketState.CONFIRMED_CHOCH_BULL, MarketState.CONFIRMED_CHOCH_BEAR]:
            return None

        cvd_z = np.nan
        if len(self.cvd_1m_deq) > 10:
            mean, std = np.mean(self.cvd_1m_deq), np.std(self.cvd_1m_deq)
            if std > 1e-9: cvd_z = (self.bar_data['1m']['cvd'] - mean) / std
        if pd.isna(cvd_z): return None

        # --- Tactical Entry: Pullback to the CHOCH-inducing zone ---
        if self.market_state == MarketState.CONFIRMED_CHOCH_BEAR and self.bearish_ob_zones:
            reversal_zone = self.bearish_ob_zones[-1] # Target the OB that caused the break
            if self.prev_price < reversal_zone['bottom'] and self.current_price >= reversal_zone['bottom']:
                # Confirm with order flow
                if cvd_z < -self.smc_cfg.CVD_Z_SCORE_THRESHOLD:
                    return Side.SELL

        if self.market_state == MarketState.CONFIRMED_CHOCH_BULL and self.bullish_ob_zones:
            reversal_zone = self.bullish_ob_zones[-1]
            if self.prev_price > reversal_zone['top'] and self.current_price <= reversal_zone['top']:
                if cvd_z > self.smc_cfg.CVD_Z_SCORE_THRESHOLD:
                    return Side.BUY
        
        return None
