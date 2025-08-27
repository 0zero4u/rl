# generator.py

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from tqdm.auto import tqdm
from typing import List, Dict, Any, Type
from ..config import SETTINGS
from ..utils import get_files_for_period
from .engine import HotCache
from .strategies import (
    PureMomentumChaser, PureMeanReversion,
    ContextAwareMomentum, ContextAwareMeanReversion
)

STRATEGIES = {
    "pure_momentum": PureMomentumChaser,
    "pure_mean_reversion": PureMeanReversion,
    "context_aware_momentum": ContextAwareMomentum,
    "context_aware_mean_reversion": ContextAwareMeanReversion,
}

def _run_simulation_for_period(period_name: str, strategy_class: Type[HotCache], strategy_name: str):
    print(f"\n--- Starting: {period_name.upper()} | Strategy: {strategy_class.__name__} ---")
    trade_files = get_files_for_period(period_name, "processed_trades")
    if not trade_files: return

    total_rows = sum(pq.ParquetFile(f).metadata.num_rows for f in trade_files)
    cache = strategy_class()
    intent_payloads: List[Dict[str, Any]] = []

    with tqdm(total=total_rows, desc=f"Processing {period_name} ({strategy_name})") as pbar:
        for trade_file in trade_files:
            try:
                for batch in pq.ParquetFile(trade_file).iter_batches(batch_size=SETTINGS.BATCH_SIZE):
                    df = batch.to_pandas()
                    if 'funding_rate' not in df.columns: df['funding_rate'] = np.nan
                    
                    for row in df.itertuples(index=False, name='Tick'):
                        cache.update(row)
                        if trigger_side := cache.check_trigger():
                            payload = cache.get_intent_payload(trigger_side, strategy_name)
                            intent_payloads.append(payload)
                    pbar.update(df.shape[0])
            except Exception as e:
                print(f"\nERROR processing file {trade_file}: {e}")

    if not intent_payloads:
        print("⚠️ No trigger events found for this strategy in this period."); return

    df = pd.json_normalize(intent_payloads, sep='.')
    output_path = SETTINGS.get_intents_path(period_name, strategy_name)
    df.to_parquet(output_path, index=False, compression=SETTINGS.PARQUET_COMPRESSION)
    print(f"✅ Saved {len(df)} intents to: {output_path}")

def generate_all_features():
    """Runs simulations for all strategies and periods, then combines the results."""
    all_payload_dfs = []
    for period in ["in_sample", "out_of_sample"]:
        for name, cls in STRATEGIES.items():
            _run_simulation_for_period(period, cls, name)
            
            # Load the generated file to combine later
            intent_path = SETTINGS.get_intents_path(period, name)
            if pd.io.common.file_exists(intent_path):
                df = pd.read_parquet(intent_path)
                df['period'] = period
                all_payload_dfs.append(df)

    if not all_payload_dfs:
        print("\n\n⚠️ No payloads were generated. No final dataset created.")
        return

    print("\n\n--- Combining all generated intent payloads ---")
    final_df = pd.concat(all_payload_dfs, ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    final_path = SETTINGS.get_combined_intents_path()
    final_df.to_parquet(final_path, index=False, compression=SETTINGS.PARQUET_COMPRESSION)
    print(f"✅ Successfully saved the final unified dataset to: {final_path}")
    print(final_df.groupby('period')['strategy_id'].value_counts())
