# src/seeding.py
import numpy as np
import pandas as pd
import re

from datetime import datetime, timedelta
from llama_cpp import Llama
from src import config, summarization

def clean_headline(text):
    """Removes leading numbers, dots, and whitespace from LLM output."""
    # Removes "1. ", "2) ", etc.
    return re.sub(r'^\d+[\.\)]\s*', '', text).strip()


def run():
    print('\nLoading model for seed data generation...')

    llm = Llama(
        model_path=config.LLAMA_MODEL_PATH,
        n_ctx=config.LLAMA_CONTEXT_SIZE,
        n_gpu_layers=-1
    )

    # Noise Thresholds
    DRIFT_NOISE = 0.03
    PRICE_NOISE = 0.01
    VOLUME_NOISE = 0.10

    sentiments = ['Positive', 'Negative', 'Neutral']
    label_map = {'Positive': '1', 'Negative': '-1', 'Neutral': '0'}

    # Initialize the modern NumPy random generator
    rng = np.random.default_rng(config.SEED)

    # 1. Get baseline stats from original data
    df_orig = pd.read_csv(config.SOURCE_FILE)
    median_open = df_orig['open'].median()
    median_vol = int(df_orig['volume'].median())

    # Use the last date as today's date
    last_date = df_orig['date'].max()

    data = []
    date_ctr = 1
    for s in sentiments:
        print(f'\nGenerating {s} headlines...')
        
        prompt = f"""[INST] <<SYS>>
        You are a financial news generator. Output ONLY raw headlines, one per line. 
        No numbers, no introduction(s), no bullet points.
        Headlines should be around 200 to 400 characters in string length.
        Each headline must be original and unique!
        <</SYS>>
        Generate {config.NUM_ROWS_PER_SENTIMENT} realistic headlines with a {s} market sentiment. [/INST]"""

        response = llm(prompt, max_tokens=2000, temperature=0.8) # Slight temp increase for variety
        lines = response['choices'][0]['text'].split('\n')

        for line in lines:

            # Convert the string to a datetime object, add the days, then format back to string
            date = (datetime.strptime(last_date, '%Y-%m-%d') + timedelta(days=date_ctr)).strftime('%Y-%m-%d')

            cleaned = clean_headline(line)
            if len(cleaned) > 25 and not cleaned.lower().startswith('here are'):

                # --- NOISE GENERATION ---
                # Create a random 'drift' between -3% and +3%
                drift = rng.uniform(1 - DRIFT_NOISE, 1 + DRIFT_NOISE)

                # Base price for this specific row
                row_open = round(median_open * drift, 2)
                row_close = round(row_open * rng.uniform(1 - PRICE_NOISE, 1 + PRICE_NOISE), 2)

                # Ensure High is actually the highest and Low is actually the lowest
                row_high = round(max(row_open, row_close) * rng.uniform(1.001, 1.005), 2)
                row_low = round(min(row_open, row_close) * rng.uniform(0.995, 0.999), 2)

                # Volume noise (+/- 10%)
                row_vol = int(median_vol * rng.uniform(1 - VOLUME_NOISE, 1 + VOLUME_NOISE))

                data.append({
                    'date': date,
                    'news': cleaned,
                    'open': row_open,
                    'high': row_high,
                    'low': row_low,
                    'close': row_close,
                    'volume': row_vol,
                    'label': label_map[s],
                })

                # Increment date by one day only if a row was added
                date_ctr += 1

    df_new = pd.DataFrame(data, columns=['date', 'news', 'open', 'high', 'low', 'close', 'volume', 'label'])
    df_new.to_csv(config.SEEDER_FILE, index=False)
    
    print(f"\n* Done! Created {len(df_new)} clean rows with realistic price noise! *")
