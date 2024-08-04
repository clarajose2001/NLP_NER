import requests
import nltk
from nltk.tokenize import word_tokenize
from nltk.chunk import ne_chunk
from nltk import pos_tag

# Step 1: Fetch the news article
def fetch_news_article(api_key):
    url = f'https://newsapi.org/v2/top-headlines?country=us&apiKey={api_key}'
    response = requests.get(url)
    print("API request status code:", response.status_code)  # Debug: print the response status code
    if response.status_code == 200:
        data = response.json()
        print("Full response data:", data)  # Debug: print the full response data
        if 'articles' in data and len(data['articles']) > 0:
            article_data = data['articles'][0]
            # Attempt to extract from multiple fields
            article = article_data.get('content') or article_data.get('description') or article_data.get('title')
            print("Fetched article content:", article)  # Debug: print the fetched article content
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
        # Tokenize and tag parts of speech
        tokens = word_tokenize(article)
        tagged_tokens = pos_tag(tokens)
        
        # Perform Named Entity Recognition
        named_entities = ne_chunk(tagged_tokens)
        
        # Extract and print named entities
        entities = set()
        for subtree in named_entities:
            if hasattr(subtree, 'label'):
                entity = " ".join([word for word, tag in subtree])
                entities.add((entity, subtree.label()))
        return entities
    else:
        print("No content to process for NLTK.")
        return set()

def main():
    # Replace 'YOUR_API_KEY' with your actual News API key
    api_key = 'd1800661ac954e10a0eeaf518c1370ac'
    article = fetch_news_article(api_key)
    
    if article:
        # NLTK setup
        nltk.download('punkt', quiet=True)
        nltk.download('maxent_ne_chunker', quiet=True)
        nltk.download('words', quiet=True)

        nltk_entities = extract_entities_nltk(article)
        print("NLTK Named Entities:")
        print(nltk_entities)
    else:
        print("No article to process.")

if __name__ == "__main__":
    main()
