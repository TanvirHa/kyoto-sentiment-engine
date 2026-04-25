import joblib
from janome.tokenizer import Tokenizer # ADD THIS

# --- YOU NEED THIS PART AGAIN ---
t = Tokenizer()

def tokenize_jp(text):
    target_pos = ['名詞', '形容詞', '動詞'] 
    tokens = []
    for token in t.tokenize(text):
        pos = token.part_of_speech.split(',')[0]
        if pos in target_pos:
            # base_form is the "root" of the word
            tokens.append(token.base_form)
    return tokens
# --------------------------------

# Now loading the brain will work because it can "see" the function above
model = joblib.load('kyoto_model.pkl')

test_sentences = [
    "京都の古い町並みが本当に綺麗でした！",
    "接客が最悪で、二度と行きません。",
    "味は普通ですが、少し高いですね。"
]

print("--- Kyoto Sentiment Engine Results ---")

for text in test_sentences:
    prediction = model.predict([text])[0]
    print(f"Review: {text}")
    print(f"Sentiment: {prediction}")
    print("-" * 30)