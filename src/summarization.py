# src/summarization.py
import pandas as pd
from llama_cpp import Llama

from src import config

def aggregate_weekly_news(data: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates news articles by week.
    """
    data['date'] = pd.to_datetime(data['date'])
    data['year_week'] = data['date'].dt.strftime('%Y-%U')
    weekly_news = data.groupby('year_week')['final_clean_text'].apply(lambda x: ' '.join(x)).reset_index()
    weekly_news.rename(columns={'final_clean_text': 'news'}, inplace=True)
    return weekly_news

def load_llama_model(model_path: str = config.LLAMA_MODEL_PATH) -> Llama:
    """
    Loads the Llama model from the specified path.
    """
    return Llama(model_path=model_path, n_ctx=config.LLAMA_CONTEXT_SIZE)

def get_mistral_response(llm: Llama, news: str, prompt: str = config.LLAMA_SUMMARIZATION_PROMPT) -> str:
    """
    Mistral v1 AI Model
    This function is designed to interact with a large language model (LLM) named Mistral,
    specifically version 1, sending it a prompt and relevant news articles, and then
    processing the model's response. It aims to return a JSON object extracted from
    the model's output.
    """

    prompt_str = f"""[INST]{prompt}News Articles: {news}[/INST]"""

    llm_resp = llm(
        prompt_str, # Prompt to send to the LlaMa
        max_tokens=200,   # Controls max response length
        temperature=0.5,  # Slight randomness for varied responses
        top_p=0.95,  # Nucleus sampling
        top_k=50,    # Limits vocabulary choice
        echo=False,
    )

    resp = llm_resp['choices'][0]['text']

    if not resp:
        print(f'Warning: News article of {len(news)} characters produced no output!')
        return ''

    return filter_response(resp.strip())

def parse_and_format_responses(weekly_news: pd.DataFrame) -> str:
    """
    Parses and formats the model's responses.
    """
    output = []
    for _, row in weekly_news.iterrows():
        output.append(f"Week {row['year_week']}:\n{row['model_response']}\n")
    return "\n".join(output)

def filter_response(text: str) -> str:
    """
    A simple filter to clean up the model's response.
    """
    return text.strip()
