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


def preprocess_text(df: pd.DataFrame) -> pd.DataFrame:
    """Applies all text preprocessing steps to the 'news' column."""
    df['final_clean_text'] = df['news'].apply(remove_special_chars).str.lower().str.strip().apply(remove_stopwords).apply(apply_porter_stemmer)
    return df
