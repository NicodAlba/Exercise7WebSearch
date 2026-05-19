import nltk
import os
import csv

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

def download_nltk_data():
    print("Checking/downloading NLTK models...")
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

def load_corpus(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: '{file_path}' not found.")
        
    reviews = []
    with open(file_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['Review'].strip(): 
                reviews.append(row['Review'].strip())
                
    print(f"Loaded {len(reviews)} reviews from CSV.")
    return reviews

def process_reviews_with_pos(reviews_list):
    print("\nProcessing reviews through the NLTK POS-Tagger...")
    processed_docs = []
    
    for text in reviews_list:
        tokens = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(tokens)
        processed_docs.append(tagged)
        
    print("All reviews have been POS-tagged successfully!")
    return processed_docs

download_nltk_data()

data_path = "reviews.csv"
my_reviews = load_corpus(data_path)

tagged_reviews = process_reviews_with_pos(my_reviews)

print("\n--- Example output for Review 1 ---")
for word, tag in tagged_reviews[0]:
    print(f"Word: {word:15} POS-Tag: {tag}")

sentiment_dict = {
    # Positives
    "tasty": "+", "good": "+", "soothing": "+", "nice": "+", "amazing": "+", 
    "delicious": "+", "perfect": "+", "friendly": "+", "superb": "+", "exquisite": "+",
    "comfortable": "+", "reasonable": "+", "popular": "+",
    # Negatives
    "rude": "-", "cold": "-", "terrible": "-", "wrong": "-", "mediocre": "-", 
    "tough": "-", "dirty": "-", "lukewarm": "-", "drab": "-"
}

def extract_opinions(tagged_docs):

    print("\nExtracting opinions based on rules...")
    extracted_opinions = []

    for doc in tagged_docs:
        length = len(doc)
        
        for i in range(length - 1):
            word1, tag1 = doc[i]
            word2, tag2 = doc[i+1]
            
            w1_lower = word1.lower()
            w2_lower = word2.lower()
            
            if w1_lower in ["not", "n't"] and tag2.startswith('JJ'):
                phrase = f"{word1} {word2}"
                extracted_opinions.append(("Negation", phrase, "-"))
                continue

            if tag1.startswith('JJ') and tag2.startswith('NN'):
                phrase = f"{word1} {word2}"
                polarity = sentiment_dict.get(w1_lower, "Neutral")
                extracted_opinions.append(("ADJ+NOUN", phrase, polarity))

            if tag1.startswith('VB') and tag2.startswith('JJ'):
                phrase = f"{word1} {word2}"
                polarity = sentiment_dict.get(w2_lower, "Neutral")
                extracted_opinions.append(("VERB+ADJ", phrase, polarity))

            if i < length - 2:
                word3, tag3 = doc[i+2]
                w3_lower = word3.lower()
                
                if tag1.startswith('RB') and tag2.startswith('JJ') and tag3.startswith('NN'):
                    phrase = f"{word1} {word2} {word3}"
                    polarity = sentiment_dict.get(w2_lower, "Neutral")
                    extracted_opinions.append(("ADV+ADJ+NOUN", phrase, polarity))
                    
                if tag1.startswith('VB') and tag3.startswith('JJ'):
                    phrase = f"{word1} {word2} {word3}"
                    polarity = sentiment_dict.get(w3_lower, "Neutral")
                    extracted_opinions.append(("VERB+#+ADJ", phrase, polarity))

    return extracted_opinions

found_opinions = extract_opinions(tagged_reviews)

print("\n=== EXTRACTED OPINIONS ===")
for rule, text, pol in found_opinions:
    if pol in ["+", "-"]:
        print(f"[{pol}] Rule: {rule:15} | Phrase: '{text}'")

total_polarized = len([op for op in found_opinions if op[2] in ['+', '-']])
print(f"\nTotal polarized opinions found: {total_polarized}")