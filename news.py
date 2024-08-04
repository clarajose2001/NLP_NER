import requests

# Replace 'YOUR_API_KEY' with your actual News API key
api_key = 'd1800661ac954e10a0eeaf518c1370ac'
url = f'https://newsapi.org/v2/top-headlines?country=us&apiKey={api_key}'

response = requests.get(url)
data = response.json()

# Get the first article's content
article = data['articles'][0]['content']

import nltk
nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')

import requests
import nltk
import ner_comparison
from nltk.tokenize import word_tokenize
from nltk.chunk import ne_chunk
from nltk import pos_tag

# Step 1: Fetch the news article
def fetch_news_article(api_key):
    url = f'https://newsapi.org/v2/top-headlines?country=us&apiKey={api_key}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if 'articles' in data and len(data['articles']) > 0:
            article = data['articles'][0].get('content')
            return article
        else:
            print("No articles found in the response.")
            return None
    else:
        print("Failed to fetch articles:", response.status_code, response.text)
        return None

# Step 2: Extract entities using NLTK
def extract_entities_nltk(article):
    if article:
        tokens = word_tokenize(article)
        tagged_tokens = pos_tag(tokens)
        named_entities = ne_chunk(tagged_tokens)
        
        entities = set()
        for subtree in named_entities:
            if hasattr(subtree, 'label'):
                entity = " ".join([word for word, tag in subtree])
                entities.add((entity, subtree.label()))
        return entities
    else:
        print("No content to process for NLTK.")
        return set()

# Step 3: Extract entities using SpaCy
def extract_entities_spacy(article):
    if article:
        nlp = ner_comparison.load('en_core_web_sm')
        doc = nlp(article)
        entities = set((ent.text, ent.label_) for ent in doc.ents)
        return entities
    else:
        print("No content to process for SpaCy.")
        return set()

# Step 4: Compare entities
def compare_entities(nltk_entities, spacy_entities):
    common_entities = nltk_entities.intersection(spacy_entities)
    nltk_only_entities = nltk_entities - spacy_entities
    spacy_only_entities = spacy_entities - nltk_entities
    
    print("Common Entities:")
    print(common_entities)
    
    print("\nEntities only found by NLTK:")
    print(nltk_only_entities)
    
    print("\nEntities only found by SpaCy:")
    print(spacy_only_entities)

def main():
    # Replace 'YOUR_API_KEY' with your actual News API key
    api_key = 'd1800661ac954e10a0eeaf518c1370ac'
    article = fetch_news_article(api_key)
    
    if article:
        # NLTK setup
        nltk.download('punkt')
        nltk.download('maxent_ne_chunker')
        nltk.download('words')

        nltk_entities = extract_entities_nltk(article)
        print("NLTK Named Entities:")
        print(nltk_entities)

        spacy_entities = extract_entities_spacy(article)
        print("\nSpaCy Named Entities:")
        print(spacy_entities)

        compare_entities(nltk_entities, spacy_entities)

if __name__ == "__main__":
    main()
