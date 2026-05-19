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