# main.py

import argparse
import gc
import pandas as pd

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src import config, utils, data_processing, summarization

def run_training_pipeline():
    """
    Executes the full model training and evaluation pipeline.
    """
    # --- 1. Setup & Data Loading ---
    utils.show_banner("STARTING MODEL TRAINING PIPELINE")
    start_time = utils.start_timer()
    utils.show_hardware_info()

    data = data_processing.load_data()
    print("Data loaded successfully.")

    # --- 2. Data Preprocessing ---
    utils.show_banner("Preprocessing Text Data")
    data = data_processing.preprocess_text(data)
    print("Text preprocessing complete.")

    # --- 3. Data Splitting ---
    utils.show_banner("Splitting Data")
    x = data['final_clean_text']
    y = data['label']

    x_train, x_temp, _, y_temp = train_test_split(
        x, y, test_size=config.TEMPORARY_DATA_SPLIT, stratify=y, random_state=config.SEED
    )
    x_val, x_test, _, _ = train_test_split(
        x_temp, y_temp, test_size=config.HALF_DATA_SPLIT, stratify=y_temp, random_state=config.SEED
    )
    print("Data split into training, validation, and test sets.")
    print(f'Training set size: {len(x_train)}')
    print(f'Validation set size: {len(x_val)}')
    print(f'Test set size: {len(x_test)}')

    # --- 4. Feature Engineering (Embeddings) ---
    utils.show_banner("Generating Word Embeddings")
    print("Placeholder for embedding generation.")

    # --- 5. Model Training and Tuning ---
    utils.show_banner("Training and Tuning Models")
    print("Placeholder for model training.")

    # --- 6. Final Model Evaluation ---
    utils.show_banner("Final Model Evaluation")
    print("Placeholder for final evaluation.")

    utils.show_timer(start_time)
    utils.show_banner("MODEL TRAINING PIPELINE COMPLETE")


def run_summarization_pipeline():
    """
    Executes the weekly news summarization pipeline using the LLM.
    """
    utils.show_banner("STARTING NEWS SUMMARIZATION PIPELINE")
    start_time = utils.start_timer()

    # --- 1. Load and Aggregate Data ---
    data = data_processing.load_data()
    weekly_news = summarization.aggregate_weekly_news(data)
    print(f"Aggregated news into {len(weekly_news)} weeks.")

    # --- 2. Load LLM ---
    llm = summarization.load_llama_model()

    # --- 3. Generate Summaries ---
    tqdm.pandas(desc="Generating weekly summaries")
    weekly_news['model_response'] = weekly_news['news'].progress_apply(
        lambda x: summarization.get_mistral_response(llm, x)
    )

    # --- 4. Parse and Display Results ---
    final_output = summarization.parse_and_format_responses(weekly_news)
    utils.show_banner("Weekly News Summarization Results")
    print(final_output)

    utils.show_timer(start_time)
    utils.show_banner("NEWS SUMMARIZATION PIPELINE COMPLETE")


if __name__ == '__main__':
    # This allows running specific parts of the project from the command line
    parser = argparse.ArgumentParser(description="Stock Market News Analysis Pipeline.")
    parser.add_argument('--pipeline', choices=['training', 'summarization', 'all'], default='all',
                        help='Which pipeline to run.')

    args = parser.parse_args()

    if args.pipeline in ['training', 'all']:
        run_training_pipeline()

    if args.pipeline in ['summarization', 'all']:
        run_summarization_pipeline()

    gc.collect()
