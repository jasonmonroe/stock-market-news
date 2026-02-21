# main.py

# =========================================
#  STOCK MARKET NEWS ANALYSIS & PREDICTOR
# =========================================

import argparse
import gc
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

from src import config, utils, data_processing, summarization, modeling, eda

def run_training_pipeline(seed_data: bool=False):
    """
    Executes the full model training and evaluation pipeline.
    """

    # --- 1. Setup & Data Loading ---
    utils.show_banner('STARTING MODEL TRAINING PIPELINE')
    start_time = utils.start_timer()
    utils.show_hardware_info()

    data = data_processing.load_data(seed_data=seed_data)
    print(
        f'\nLoaded {len(data)} rows of data from {config.DATASET_FILE}.'
    )
    print('Data loaded successfully.')

    # --- 2. Data Preprocessing ---
    utils.show_banner('Preprocessing Text Data')
    data = data_processing.preprocess_text(data)
    print('Text preprocessing complete.')

    # --- 3. Data Splitting ---
    # ===========================================
    #  CREATE TRAINING, VALIDATION, TESTING DATA
    # ===========================================
    #
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    #
    # Training Data ~ 80%
    # Validation Data ~ 10%
    # Testing Data ~ 10%
    #
    # Define input (X) and target labels (y)

    # INDEPENDENT VARIABLES aka features
    utils.show_banner("Splitting Data")
    df_features = data['final_clean_text']
    df_target = data['label'] # Dependent variable

    # Split data into 80% train, 20% temporary (to be split into validation & test)
    x_train, x_temp, y_train, y_temp = train_test_split(
        df_features,
        df_target,
        test_size=config.TEMPORARY_DATA_SPLIT,
        stratify=df_target,
        random_state=config.SEED
    )

    # Further split temporary set into 10% validation, 10% test
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=config.HALF_DATA_SPLIT,
        stratify=y_temp,
        random_state=config.SEED
    )

    print("Data split into training, validation, and test sets.")
    print(f'Training set size: {len(x_train)}')
    print(f'Validation set size: {len(x_val)}')
    print(f'Test set size: {len(x_test)}')

    # --- 4. Feature Engineering (Embeddings) ---
    utils.show_banner("Generating Word Embeddings")
    print("Placeholder for embedding generation.")

    x_train_w2v, x_val_w2v, x_test_w2v = modeling.create_w2v_embeddings(x_train, x_val, x_test, data['final_clean_text'])
    x_train_glove, x_val_glove, x_test_glove = modeling.create_glove_embeddings(x_train, x_val, x_test)
    x_train_st, x_val_st, x_test_st = modeling.create_st_embeddings(x_train, x_val, x_test)

    # --- 5. Model Training and Tuning ---
    utils.show_banner("Training and Tuning Models")
    print("Placeholder for model training.")

    # Run the full modeling process to determine which is the best model and predictor.
    final_model, final_predictor, model_title = modeling.run_full_modeling_process(
        x_train_w2v, y_train, x_val_w2v, y_val, x_test_w2v,
        x_train_glove, x_val_glove, x_test_glove,
        x_train_st, x_val_st, x_test_st
    )

    # --- 6. Final Model Evaluation ---
    utils.show_banner("Final Model Evaluation")
    print("Placeholder for final evaluation.")

    final_perf = modeling.model_performance_classification_sklearn(final_model, final_predictor, y_test)
    utils.show_performance(final_perf, f'{model_title} - Final Test Performance')

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
    print(f'Aggregated news into {len(weekly_news)} weeks.')

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

def run_eda_pipeline():
    """
    Executes the Exploratory Data Analysis pipeline to generate and show plots.
    """
    utils.show_banner("STARTING EXPLORATORY DATA ANALYSIS (EDA) PIPELINE")
    start_time = utils.start_timer()

    data = data_processing.load_data()
    data = data_processing.preprocess_text(data)
    print("Data loaded and preprocessed.")

    # --- Generate and Show Plots ---
    print("\nShowing Labeled Barchart for Sentiment...")
    chart_df = data.copy()
    chart_df['label'] = chart_df['label'].replace(config.LABEL_SENTIMENT)
    eda.labeled_barchart(chart_df, 'label')

    print("\nShowing Word Count Distribution...")
    data['word_count'] = data['news'].apply(lambda x: len(str(x).split()))
    eda.plot_word_count_distribution(data, 'word_count')

    print("\nShowing Correlation Matrix...")
    numerical_data = data.select_dtypes(include=['number'])
    correlation_matrix = numerical_data.corr()
    eda.show_correlation_matrix(correlation_matrix)

    print("\nShowing Top Word Frequency...")
    bow_vec = CountVectorizer(max_features=config.VEC_MAX_SIZE)
    bow_vec.fit_transform(data['final_clean_text'])
    words = bow_vec.get_feature_names_out()
    word_counts = bow_vec.transform(data['final_clean_text']).toarray().sum(axis=0)
    eda.plot_top_word_freq(words, word_counts)

    utils.show_timer(start_time)
    utils.show_banner("EDA PIPELINE COMPLETE")

if __name__ == '__main__':

    main_start_time = utils.start_timer()
    run_id = utils.get_run_id()
    print(f'\n#--- {run_id} | START PROGRAM ---#')

    parser = argparse.ArgumentParser(description='Stock Market News Analysis Pipeline.')

    # Existing pipeline argument
    parser.add_argument(
        '--pipeline',
        choices=['training', 'summarization', 'eda', 'all'],
        default='all',
        help='Which pipeline to run: training, summarization, eda, or all.',
    )

    # NEW: Add the seed flag
    parser.add_argument(
        '--seed',
        action='store_true',  # This makes it a boolean flag (True if present, False if not)
        help='Use this flag to generate/load synthetic seed data.'
    )

    args = parser.parse_args()

    # Pass args.seed to the training pipeline
    if args.pipeline in ['training', 'all']:
        run_training_pipeline(seed_data=args.seed)

    if args.pipeline in ['summarization', 'all']:
        run_summarization_pipeline()

    if args.pipeline == 'eda':
        run_eda_pipeline()

    gc.collect()
    utils.show_timer(main_start_time)

    print(f'\n#--- {run_id} | END PROGRAM ---#\n')