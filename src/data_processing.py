# src/data_processing.py

import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from src import config


def load_data(path: str = config.CSV_FILE) -> pd.DataFrame:
    """Loads data from a CSV file."""
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    return pd.read_csv(path)


def remove_special_chars(text: str) -> str:
    """Removes special characters from a string."""
    pattern = '[^A-Za-z0-9]+'
    return ''.join(re.sub(pattern, ' ', text))


def remove_stopwords(text: str) -> str:
    """Removes English stopwords from a string."""
    words = text.split()
    new_text = ' '.join([word for word in words if word not in stopwords.words('english')])
    return new_text


def apply_porter_stemmer(text: str) -> str:
    """Applies Porter Stemmer to a string."""
    ps = PorterStemmer()
    words = text.split()
    return ' '.join([ps.stem(word) for word in words])


def _get_optimized_preprocessor():
    """
    Creates and returns an optimized text preprocessing function.
    This avoids re-initializing objects and lists on every call.
    """
    # Initialize objects once
    stemmer = PorterStemmer()
    # Use a set for O(1) average time complexity lookups
    stop_words = set(stopwords.words('english'))
    pattern = re.compile('[^A-Za-z0-9]+')

    def preprocess(text: str) -> str:
        # 1. Remove special characters and convert to lowercase
        cleaned_text = re.sub(pattern, ' ', text).lower().strip()
        # 2. Remove stopwords and apply stemmer
        words = [stemmer.stem(word) for word in cleaned_text.split() if word not in stop_words]
        return ' '.join(words)

    return preprocess


def preprocess_text(df: pd.DataFrame) -> pd.DataFrame:
    """Applies all text preprocessing steps to the 'news' column."""
    optimized_processor = _get_optimized_preprocessor()
    df['final_clean_text'] = df['news'].apply(optimized_processor)
    return df
