# labeler.py

import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from ..config import SETTINGS
from ..data.processor import create_bars_from_trades

def apply_triple_barrier_labeling():
    """Loads intents, prepares bar data, applies triple barrier, and saves the result."""
    intents_path = SETTINGS.get_combined_intents_path()
    if not os.path.exists(intents_path):
        raise FileNotFoundError(f"Intents file not found at '{intents_path}'. Run feature generation first.")
        
    events_df = pd.read_parquet(intents_path)
    print(f"Loaded {len(events_df):,} trade intents from '{intents_path}'")

    bars_df = create_bars_from_trades()
    print("\n--- Applying Triple Barrier Labeling ---")
    
    labeled_events = pd.merge_asof(
        events_df.sort_values('timestamp'),
        bars_df[['timestamp', 'close', 'atr']].sort_values('timestamp'),
        on='timestamp', direction='backward'
    ).rename(columns={'close': 'entry_price'}).dropna(subset=['entry_price', 'atr'])

    outcomes = []
    bars_indexed = bars_df.set_index('timestamp')
    cfg = SETTINGS.labeling

    for event in tqdm(labeled_events.itertuples(), total=len(labeled_events), desc="Labeling events"):
        side = 1 if event.side == 'BUY' else -1
        profit_target = event.entry_price + (side * event.atr * cfg.PROFIT_TAKE_ATR)
        stop_loss = event.entry_price - (side * event.atr * cfg.STOP_LOSS_ATR)
        time_barrier = event.timestamp + pd.Timedelta(minutes=cfg.TIME_LIMIT_MINUTES)
        future_bars = bars_indexed.loc[event.timestamp:time_barrier]
        
        label, outcome_ts = 0, time_barrier
        for bar_ts, bar in future_bars.iterrows():
            if (side == 1 and bar.low <= stop_loss) or (side == -1 and bar.high >= stop_loss):
                label, outcome_ts = -1, bar_ts; break
            if (side == 1 and bar.high >= profit_target) or (side == -1 and bar.low <= profit_target):
                label, outcome_ts = 1, bar_ts; break
        
        outcomes.append({'label': label, 'outcome_timestamp': outcome_ts, 'profit_target': profit_target, 'stop_loss': stop_loss})
        
    outcomes_df = pd.DataFrame(outcomes, index=labeled_events.index)
    final_df = pd.concat([labeled_events, outcomes_df], axis=1)

    output_path = SETTINGS.get_labeled_data_path()
    final_df.to_parquet(output_path, index=False, compression=SETTINGS.PARQUET_COMPRESSION)

    print("\n\n--- Labeling Complete ---")
    print(f"✅ Saved labeled dataset to: {output_path}")
    print("\n--- Label Distribution ---")
    print(final_df['label'].value_counts(normalize=True).apply("{:.2%}".format))
