import pandas as pd
import numpy as np
import re
import emoji
import contractions
import nltk
from nltk.corpus import stopwords
import spacy
import nlpaug.augmenter.word as naw
from pathlib import Path

# Setup
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
nlp = spacy.load('en_core_web_sm')
output_dir = Path('outputs')
output_dir.mkdir(exist_ok=True)

# Load dataset
try:
    df = pd.read_csv('Tweets.csv')
except FileNotFoundError:
    raise FileNotFoundError('Dataset not found at Tweets.csv.')

# Map sentiments
sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
df['label'] = df['airline_sentiment'].map(sentiment_map)

# Text cleaning
def clean_tweet(tweet):
    if not isinstance(tweet, str) or not tweet.strip():
        return ''
    tweet = emoji.demojize(tweet, delimiters=(' ', ' '))
    tweet = contractions.fix(tweet)
    tweet = re.sub(r'http\\S+|@\\w+|#\\w+', '', tweet)
    tweet = re.sub(r'[^a-zA-Z\\s]', '', tweet)
    doc = nlp(tweet.lower())
    tweet = [token.lemma_ for token in doc if token.text not in stop_words and token.text.strip()]
    return ' '.join(tweet)

df['cleaned_text'] = df['text'].apply(clean_tweet)

# Data augmentation
aug = naw.SynonymAug(aug_p=0.3)
def augment_data(df, label, n_samples):
    df_subset = df[df['label'] == label]
    augmented_texts = []
    augmented_labels = []
    for _ in range(n_samples):
        sample = df_subset.sample(1)
        text = sample['cleaned_text'].values[0]
        if text.strip():
            try:
                aug_text = aug.augment(text)[0]
                augmented_texts.append(aug_text)
                augmented_labels.append(label)
            except Exception:
                continue
    return pd.DataFrame({
        'cleaned_text': augmented_texts,
        'label': augmented_labels,
        'text': augmented_texts,
        'airline': sample['airline'].values[0]
    })

aug_positive = augment_data(df, 2, 1000)
aug_neutral = augment_data(df, 1, 1000)
df = pd.concat([df, aug_positive, aug_neutral], ignore_index=True)

# Aspect extraction
def extract_aspects(tweet):
    if not tweet.strip():
        return ['other']
    doc = nlp(tweet)
    aspects = []
    for token in doc:
        if token.pos_ == 'NOUN':
            for child in token.children:
                if child.pos_ == 'ADJ':
                    aspects.append(f'{token.text} ({child.text})')
    return aspects if aspects else ['other']

df['aspects'] = df['cleaned_text'].apply(extract_aspects)

# Save processed data
df.to_csv(output_dir / 'processed_data.csv', index=False)
print("Processed data saved to outputs/processed_data.csv")