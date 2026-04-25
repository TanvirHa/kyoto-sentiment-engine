from janome.tokenizer import Tokenizer

# Initialize the tokenizer once here
t = Tokenizer()

def tokenize_jp(text):
    target_pos = ['名詞', '形容詞', '動詞'] 
    tokens = []
    for token in t.tokenize(text):
        pos = token.part_of_speech.split(',')[0]
        if pos in target_pos:
            tokens.append(token.base_form)
    return tokens