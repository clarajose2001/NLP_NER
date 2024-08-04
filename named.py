import spacy

# Load the SpaCy model
def extract_entities_spacy(article):
    nlp = spacy.load('en_core_web_sm')  # Load the SpaCy English model
    doc = nlp(article)  # Process the text
    entities = set((ent.text, ent.label_) for ent in doc.ents)  # Extract entities
    return entities

# Example usage
if __name__ == "__main__":
    article = "Apple Inc. is looking at buying U.K. startup for $1 billion. The CEO Tim Cook is expected to announce the acquisition next week."
    spacy_entities = extract_entities_spacy(article)
    print("SpaCy Named Entities:")
    print(spacy_entities)
