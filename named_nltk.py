import nltk
from nltk.tokenize import word_tokenize
from nltk.chunk import ne_chunk
from nltk import pos_tag

# Extract entities using NLTK
def extract_entities_nltk(article):
    # Tokenize the article text
    tokens = word_tokenize(article)
    # Perform POS tagging
    tagged_tokens = pos_tag(tokens)
    # Perform Named Entity Recognition
    named_entities = ne_chunk(tagged_tokens)
    
    # Extract and return entities
    entities = set()
    for subtree in named_entities:
        if hasattr(subtree, 'label'):
            entity = " ".join([word for word, tag in subtree])
            entities.add((entity, subtree.label()))
    return entities

# Example usage
if __name__ == "__main__":
    # Sample article text
    article = "Apple Inc. is looking at buying U.K. startup for $1 billion. The CEO Tim Cook is expected to announce the acquisition next week."
    
    # Ensure necessary NLTK data packages are downloaded
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)
    
    # Extract and print entities using NLTK
    nltk_entities = extract_entities_nltk(article)
    print("NLTK Named Entities:")
    print(nltk_entities)
