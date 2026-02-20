# src/modeling.py

import os
import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec, KeyedVectors
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from scipy.stats import randint

from src import config, utils

# --- Embedding Functions ---

def get_vector_columns(vector_size: int) -> list:
    return ['Feature ' + str(i) for i in range(vector_size)]

def calculate_mean_word_vectors(doc: str, word_vector_dict: dict, vocab: set, vector_size: int) -> np.ndarray:
    rng = np.random.default_rng(config.SEED)
    feature_vector = rng.uniform(-0.01, 0.01, vector_size)
    doc_list = [word for word in doc.split() if word in vocab]

    if doc_list:
        feature_vector = np.mean([word_vector_dict[word] for word in doc_list], axis=0)

    return feature_vector

def encode_embeddings(x_data: pd.Series, word_vector_dict: dict, vocab: set, vector_size: int) -> pd.DataFrame:
    vector_cols = get_vector_columns(vector_size)
    # Ensure input is string
    embeddings = x_data.apply(lambda doc: calculate_mean_word_vectors(str(doc), word_vector_dict, vocab, vector_size)).tolist()

    return pd.DataFrame(embeddings, columns=vector_cols)

def create_w2v_embeddings(x_train, x_val, x_test, full_text_series):
    word_list = [str(item).split(' ') for item in full_text_series.values]
    w2v_model = Word2Vec(
        sentences=word_list,
        vector_size=config.W2V_VECTOR_SIZE,
        window=5,
        min_count=2,
        workers=4,
        sg=1,
        epochs=config.W2V_EPOCH_CNT,
        seed=config.SEED
    )
    vocabulary = set(w2v_model.wv.key_to_index.keys())
    w2v_word_vector_dict = {word: w2v_model.wv[word] for word in vocabulary}

    x_train_emb = encode_embeddings(x_train, w2v_word_vector_dict, vocabulary, config.W2V_VECTOR_SIZE)
    x_val_emb = encode_embeddings(x_val, w2v_word_vector_dict, vocabulary, config.W2V_VECTOR_SIZE)
    x_test_emb = encode_embeddings(x_test, w2v_word_vector_dict, vocabulary, config.W2V_VECTOR_SIZE)

    return x_train_emb, x_val_emb, x_test_emb

def create_glove_embeddings(x_train, x_val, x_test):
    if not os.path.exists(config.W2V_FILE_NAME):
        print(f"GloVe model not found at {config.W2V_FILE_NAME}. Skipping GloVe.")
        return None, None, None

    try:
        gloVe_model = KeyedVectors.load_word2vec_format(config.W2V_FILE_NAME, binary=False)
    except Exception as e:
        print(f"Error loading GloVe: {e}")
        return None, None, None

    vocabulary = set(gloVe_model.index_to_key)
    gloVe_word_vector_dict = {word: gloVe_model[word] for word in vocabulary}

    x_train_emb = encode_embeddings(x_train, gloVe_word_vector_dict, vocabulary, config.GLOVE_VECTOR_SIZE)
    x_val_emb = encode_embeddings(x_val, gloVe_word_vector_dict, vocabulary, config.GLOVE_VECTOR_SIZE)
    x_test_emb = encode_embeddings(x_test, gloVe_word_vector_dict, vocabulary, config.GLOVE_VECTOR_SIZE)

    return x_train_emb, x_val_emb, x_test_emb

def create_st_embeddings(x_train, x_val, x_test):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'

    st_model = SentenceTransformer(config.SENTENCE_TRANSFER_MODEL, device=device)

    def encode(data):
        return st_model.encode(data.tolist(), show_progress_bar=True)

    x_train_emb = pd.DataFrame(encode(x_train))
    x_val_emb = pd.DataFrame(encode(x_val))
    x_test_emb = pd.DataFrame(encode(x_test))

    return x_train_emb, x_val_emb, x_test_emb

# --- Modeling Functions ---

def rf_base_model():
    return RandomForestClassifier(
        max_depth=config.BASE_DEPTH_MAX,
        min_samples_leaf=config.BASE_LEAF_MIN,
        min_samples_split=config.BASE_SPLIT_MIN,
        n_estimators=config.BASE_ESTIMATOR_CNT,
        random_state=config.SEED
    )

def rf_tuned_model():
    return RandomForestClassifier(
        max_depth=4,
        min_samples_leaf=40,
        min_samples_split=80,
        n_estimators=500,
        max_features='sqrt',
        random_state=config.SEED
    )

def model_performance_classification_sklearn(model, predictors, target):
    pred = model.predict(predictors)
    acc = accuracy_score(target, pred)
    recall = recall_score(target, pred, average='weighted')
    precision = precision_score(target, pred, average='weighted')
    f1 = f1_score(target, pred, average='weighted')

    return pd.DataFrame({
        'Accuracy': [acc],
        'Recall': [recall],
        'Precision': [precision],
        'F1': [f1]
    })

def tune_model(mod, x_train, y_train):
    parameters = {
        "max_depth": [3, 4, 5, 6],
        "min_samples_leaf": randint(20, 60),
        "min_samples_split": randint(40, 100),
        "n_estimators": [300, 500, 700],
        "max_features": ["sqrt", 0.4]
    }

    random_search = RandomizedSearchCV(
        estimator=mod,
        param_distributions=parameters,
        n_iter=config.CV_RANDOM_SEARCH_CNT,
        scoring='f1_weighted',
        cv=StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.SEED),
        n_jobs=config.USE_ALL_PROCS,
        random_state=config.SEED
    )
    random_search.fit(x_train, y_train)
    return random_search.best_estimator_

def run_full_modeling_process(x_train_w2v, y_train, x_val_w2v, y_val, x_test_w2v,
                              x_train_glove, x_val_glove, x_test_glove,
                              x_train_st, x_val_st, x_test_st):

    # 1. Train Base Models
    print("\n--- Training Base Models ---")
    print("Training Base W2V Model...")
    w2v_base = rf_base_model().fit(x_train_w2v, y_train)
    w2v_perf = model_performance_classification_sklearn(w2v_base, x_val_w2v, y_val)
    utils.show_performance(w2v_perf, "W2V Base Validation")

    if x_train_glove is not None:
        print("Training Base GloVe Model...")
        glove_base = rf_base_model().fit(x_train_glove, y_train)
        glove_perf = model_performance_classification_sklearn(glove_base, x_val_glove, y_val)
        utils.show_performance(glove_perf, "GloVe Base Validation")

    print("Training Base ST Model...")
    st_base = rf_base_model().fit(x_train_st, y_train)
    st_perf = model_performance_classification_sklearn(st_base, x_val_st, y_val)
    utils.show_performance(st_perf, "ST Base Validation")

    # 2. Tune Models
    print("\n--- Tuning Models ---")
    print("Tuning W2V Model (this may take a while)...")
    w2v_tuned = tune_model(rf_tuned_model(), x_train_w2v, y_train)
    w2v_tuned_perf = model_performance_classification_sklearn(w2v_tuned, x_val_w2v, y_val)
    utils.show_performance(w2v_tuned_perf, "W2V Tuned Validation")

    # For this project, we select the Tuned W2V model as the best performer
    best_model = w2v_tuned
    best_predictor_test = x_test_w2v
    title = "Word2Vec Tuned Model"

    return best_model, best_predictor_test, title
