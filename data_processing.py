import pandas as pd
import re
import string

def load_data(path):
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    df = df[['Score', 'Summary', 'Text']]
    df.dropna(inplace=True)
    return df

def get_sentiment(score):
    if score >= 4:
        return "positive"
    elif score == 3:
        return "neutral"
    else:
        return "negative"

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text