import stanza
import os
import csv

def download_models():
    print("Checking/downloading English models...")
    stanza.download('en')

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

def process_reviews_with_pos(reviews_list, pipeline):
    print("\nProcessing reviews through the POS-Tagger...")
    processed_docs = []
    
    for text in reviews_list:
        doc = pipeline(text)
        processed_docs.append(doc)
        
    print("All reviews have been POS-tagged successfully!")
    return processed_docs

download_models()

print("\nInitializing Stanza pipeline...")
nlp_pipeline = stanza.Pipeline(lang='en', processors='tokenize,pos')

data_path = "reviews.csv"
my_reviews = load_corpus(data_path)

tagged_reviews = process_reviews_with_pos(my_reviews, nlp_pipeline)

print("\n--- Example output for Review 1 ---")
for sentence in tagged_reviews[0].sentences:
    for word in sentence.words:
        print(f"Word: {word.text:15} POS-Tag: {word.pos}")