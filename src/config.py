# src/config.py

# --- Constants ---

# Files
FILE_PATH = 'data/'
FILE_NAME = 'source_data.csv'
CSV_FILE = FILE_PATH + FILE_NAME
W2V_FILE_NAME = 'models/glove.6B.100d.txt.word2vec'

# General
COMMON_WORD_CNT = 20
KBYTE = 1024  # Corrected from 1042
MSEC = 1000
SECS_IN_MIN = 60
SEED = 42

# Label Sentiments
# -1 = Negative sentiment, 0 = neutral, 1 = positive
LABEL_NEGATIVE = 'Negative'
LABEL_NEUTRAL = 'Neutral'
LABEL_POSITIVE = 'Positive'
LABEL_NEGATIVE_INT = -1
LABEL_NEUTRAL_INT = 0
LABEL_POSITIVE_INT = 1
LABEL_SENTIMENT = {
    LABEL_NEGATIVE_INT: LABEL_NEGATIVE,
    LABEL_NEUTRAL_INT: LABEL_NEUTRAL,
    LABEL_POSITIVE_INT: LABEL_POSITIVE
}

# Data Splits
HALF_DATA_SPLIT = 0.5
TEMPORARY_DATA_SPLIT = 0.2
TRAINING_DATA_SPLIT = 0.8

# Models
FITTING_GAP_THRESH = 0.05
FITTING_SCORE_THRESH = 0.75

# Model: Base Parameters
BASE_ESTIMATOR_CNT = 300
BASE_DEPTH_MAX = 15
BASE_LEAF_MIN = 6
BASE_SPLIT_MIN = 30

# Model: Tuned Parameters
TUNED_ESTIMATOR_CNT = 500
TUNED_DEPTH_MAX = 7
TUNED_LEAF_MIN = 15
TUNED_SPLIT_MIN = 30

CV_FOLDS = 5
CV_RANDOM_SEARCH_CNT = 25
USE_ALL_PROCS = -1
SENTENCE_TRANSFER_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

# Llama Model
LLAMA_MODEL_PATH = 'models/mistral-7b-instruct-v0.1.q4_K_M.gguf'
LLAMA_CONTEXT_SIZE = 4096

# LLM Prompt for Summarization
LLAMA_SUMMARIZATION_PROMPT = """
Role: You are an expert financial analyst specializing in stock market news analysis.

### **Task:**
Given a set of **news headlines**, extract **positive** and **negative** financial events that could impact stock prices.

### **Instructions:**
1. **Analyze each news headline carefully.**
2. **Identify the company, industry, or entity mentioned.**
3. **Determine the key event or action** (e.g., earnings report, product launch, legal trouble).
4. **Assess the likely impact** on stock prices:
   - **Positive News:** Events likely to **increase** stock prices.
   - **Negative News:** Events likely to **decrease** stock prices.
5. **STRICT OUTPUT FORMAT:**
   - Return **ONLY** a **valid JSON object**.
   - **DO NOT** include any introductory text, explanations, greetings, or comments.
   - The output must **begin and end with `{}`** and follow valid JSON syntax.

### **Output Format (STRICTLY ENFORCED)**:
{ "Positive News": [ "List of positive events" ], "Negative News": [ "List of negative events" ] }
"""

# Vectors
GLOVE_VECTOR_SIZE = 100
VEC_MAX_SIZE = 1000
W2V_EPOCH_CNT = 40
W2V_VECTOR_SIZE = 300
