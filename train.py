import pandas as pd
import joblib
from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# --- A. SETUP THE JAPANESE TOOL ---
# This breaks sentences into words since Japanese has no spaces.
t = Tokenizer()

def tokenize_jp(text):
    # We only want to keep these parts of speech
    target_pos = ['名詞', '形容詞', '動詞'] 
    
    tokens = []
    for token in t.tokenize(text):
        # Extract the part of speech (pos)
        pos = token.part_of_speech.split(',')[0]
        
        if pos in target_pos:
            # We use 'base_form' so that 'was delicious' and 'delicious' 
            # are treated as the exact same word (美味しい)
            tokens.append(token.base_form)
            
    return tokens

# --- B. LOAD YOUR DATA ---
print("Loading data...")
df = pd.read_csv("reviews.csv", encoding='utf-8')

# --- C. BUILD THE BRAIN (The Pipeline) ---
# TfidfVectorizer: Converts Japanese words into math numbers.
# MultinomialNB: A popular algorithm for guessing sentiment.
model = Pipeline([
    ('vectorizer', TfidfVectorizer(tokenizer=tokenize_jp)),
    ('classifier', MultinomialNB())
])

# --- D. TRAIN THE MODEL ---
print("Training the Kyoto Sentiment Engine...")
model.fit(df['review'], df['sentiment'])

# --- E. SAVE FOR LATER ---
joblib.dump(model, 'kyoto_model.pkl')
print("Done! Your model is saved as 'kyoto_model.pkl'.")