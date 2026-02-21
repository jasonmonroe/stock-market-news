# Stock Market News Sentiment Analysis & Summarization

This project analyzes daily stock market news to predict sentiment (Positive, Negative, Neutral) and its potential impact on stock prices. It includes a complete pipeline from data preprocessing and feature engineering to model training, tuning, and evaluation. Additionally, it features a component for weekly news summarization using a Large Language Model (Llama 2).

## Project Structure

```
stock-market-news/
├── data/
├── models/
├── notebooks/
├── reports/
├── src/
│   ├── config.py           # All constants and configurations
│   ├── data_processing.py  # Data loading and cleaning functions
│   ├── eda.py              # Exploratory data analysis and plotting
│   ├── modeling.py         # Model training, tuning, and evaluation
│   ├── summarization.py    # LLM-based news summarization
│   └── utils.py            # Helper functions (timers, banners, etc.)
├── .gitignore
├── main.py                 # Main script to run pipelines
├── README.md
└── requirements.txt
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd stock-market-news
    ```

2.  **Create and activate a virtual environment:**
    ***Note: Use Python v3.12.x or lower***
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Special Installation for Apple Silicon (M1/M2/M3 Macs):**
    The `llama-cpp-python` library requires a special compilation flag to enable GPU support on Apple Silicon. Run the following command *after* activating your virtual environment:
    ```sh
    export CMAKE_ARGS="-DGGML_METAL=on"
    pip install llama-cpp-python
    ```

5.  **Download Data/Models:**
    - Place the `source_data.csv` file in the `data/` directory.
    - Place the `glove.6B.100d.txt.word2vec` file in the `models/` directory.
    - If this repo doesn't have `glove.6B.100d.txt.word2vec` you can download it at https://www.kaggle.com/datasets/danielwillgeorge/glove6b100dtxt
    - Download the LLM model by running this command from the project root:
      ```bash
      curl -L https://huggingface.co/MaziyarPanahi/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/Mistral-7B-Instruct-v0.1.Q4_K_M.gguf -o models/mistral-7b-instruct-v0.1.Q4_K_M.gguf
      ```

## How to Run

The main script `main.py` can run different parts of the project using the `--pipeline` argument. All commands should be run from the project root after activating the virtual environment.

*Note: It is recommended to use the explicit path to the virtual environment's Python interpreter (`./venv/bin/python`) to ensure the correct dependencies are used, especially on macOS and Linux.*

-   **Run the entire pipeline (training and summarization):**
    This is the default behavior if no `--pipeline` flag is provided.
    ```bash
    ./venv/bin/python main.py
    ```
    
    If you want to generate seeded data to enhance the experience, run this command:
- ```bash
    ./venv/bin/python main.py --seed
    ```
- 

-   **Run only the model training pipeline:**
    ```bash
    ./venv/bin/python main.py --pipeline training
    ```

-   **Run only the news summarization pipeline:**
    ```bash
    ./venv/bin/python main.py --pipeline summarization
    ```

-   **Run only the EDA (Exploratory Data Analysis) pipeline:**
    ```bash
    ./venv/bin/python main.py --pipeline eda
    ```

-   **Adding Seeder Data:**
    You can add the `--seed` flag to any of the above commands to merge the source data with the seeder data.
    -   *Example with the default (all) pipeline:*
        ```bash
        ./venv/bin/python main.py --seed
        ```
    -   *Example with a specific pipeline:*
        ```bash
        ./venv/bin/python main.py --pipeline training --seed
        ```

# Note: Do **not** be alarmed by "low" scores.  The dataset has only 350 rows and is only used for demo purposes.
