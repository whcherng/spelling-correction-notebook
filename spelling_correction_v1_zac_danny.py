# add in tokenize at open_file, change vocabulary into correct_word, build_models change to build_ngram_models, candidate/s to suggestion/s  

import re
from nltk.tokenize import word_tokenize
from nltk import edit_distance, bigrams
from nltk.probability import FreqDist, ConditionalFreqDist

# Open and clean corpus
def open_file(file_path):
    with open(file_path, 'r', encoding="utf-8") as f:
        text = f.read().lower()
    # Separate punctuation, numbers and spaces/tabs from words and clean it
    text = re.sub(r"([.,!?'])", r" \1 ", text)  # Add spaces around punctuation
    text = re.sub(r"[^a-zA-Z'.,!? ]", " ", text)  # Remove numbers/symbols
    text = re.sub(r"\s+", " ", text)  # Collapse multiple spaces

    tokens = word_tokenize(text) # Tokenize the text
    return tokens

# Build language models
def build_ngram_models(words):
    unigram_fd = FreqDist(words)
    bigram_cfd = ConditionalFreqDist()
    for prev, curr in bigrams(words):
        bigram_cfd[prev][curr] += 1
    return unigram_fd, bigram_cfd

# Generate suggestions with prioritization
def generate_suggestions(word, correct_word, max_distance=2):
    suggestions = []
    for suggestion in correct_word:
        distance = edit_distance(word, suggestion, transpositions=True)
        if distance <= max_distance:
            suggestions.append((suggestion, distance))
    # Sort by edit distance first
    return [c[0] for c in sorted(suggestions, key=lambda x: x[1])][:100]

# Calculate the probability with corpus validation
def calc_prob(word, prev_word, unigram_fd, bigram_cfd, vocab_size):
    # Check if word exists in corpus file
    if word not in unigram_fd:
        return 0.0  # Treat OOV words as invalid
    
    # Unigram probability
    unigram_prob = unigram_fd[word] / unigram_fd.N()
    
    # Bigram probability if context exists
    if prev_word and prev_word in bigram_cfd:
        bigram_prob = (bigram_cfd[prev_word][word] + 1) / (unigram_fd[prev_word] + vocab_size)
    else:
        bigram_prob = unigram_prob
    
    return 0.1 * unigram_prob + 0.9 * bigram_prob

# Improve correction logic
def correct_spell(statement, unigram_fd, bigram_cfd, vocab_size, correct_word):
    words = statement.lower().split()
    results = []
    
    # Use top 10% most frequent words as threshold
    common_words = {word for word, _ in unigram_fd.most_common(int(len(unigram_fd)*0.3))} # 30% top freq word
    #common_words = {word for word, _ in unigram_fd.most_common(int(len(unigram_fd)*0.1))}

    for i, word in enumerate(words):
        # Skip punctuation-only tokens
        if re.fullmatch(r"[.,!?']+", word):
            continue
            
        # Get previous word context
        prev_word = words[i-1] if i > 0 and words[i-1] in unigram_fd else None
        
        # Generate suggestions if:
        # 1. Word is OOV (not in correct_word), or
        # 2. Word is rare (not in top 10% frequent words)
        if word not in unigram_fd or word not in common_words:
            suggestions = generate_suggestions(word, correct_word)
            if suggestions:
                scored = []
                for suggestion in suggestions:
                    # Skip suggestion if same as original
                    if suggestion == word:
                        continue
                        
                    edit_dist = edit_distance(word, suggestion, transpositions=True)
                    suggestion_prob = calc_prob(suggestion, prev_word, unigram_fd, bigram_cfd, vocab_size)
                    error_prob = 0.7 ** edit_dist  # Less aggressive error model
                    score = suggestion_prob * error_prob
                    scored.append((suggestion, suggestion_prob, edit_dist, score))
                
                if scored:
                    # Sort by score and distance
                    top_suggestions = sorted(scored, key=lambda x: (-x[3], x[2]))[:5]
                    #top_suggestions = sorted(scored, key=lambda x: (-x[3], x[2]))[:5]
                    results.append((word, top_suggestions))
    
    return results

# Format and display results
def display_results(statement, correction_results):
    words = statement.split()
    output = []
    corrected_statements = []
    
    # Create mapping of incorrect words to their corrections
    #corrections_map = {res[0]: res[1] for res in correction_results}
    corrections_map = {res[0]: res[1][0][0] for res in correction_results}

    for word in words:
        if word.lower() in corrections_map:
            output.append(f"*{word}*")  # Highlight incorrect word
            corrected_statements.append(corrections_map[word.lower()]) # To add corrected word
        else:
            output.append(word)
            corrected_statements.append(word)
    
    print("\nOriginal statement with Errors Highlighted:")
    print(" ".join(output))
    
    print("\nCorrection Suggestions:")
    for incorrect_word, suggestions in correction_results:
        print(f"\nIncorrect word: {incorrect_word}")
        print("Top 5 suggestions:")
        for i, (word, prob, distance, score) in enumerate(suggestions, 1):
            print(f"{i}. {word} (Prob: {prob:.6f}, Distance: {distance}, Score: {score:.6f})")

    print("\nCorrected statements: ")
    print(" ".join(corrected_statements))
    print("\n")

# Main function
def main():
    # Load corpus and build models
    corpus_file = 'D:/MASTER/NLP/Assignment/output-COVID19.txt'  # Update with your corpus path
    words = open_file(corpus_file)
    unigram_fd, bigram_cfd = build_ngram_models(words)
    vocab_size = len(unigram_fd)
    correct_word = list(unigram_fd.keys())
    
    # Example statement
    #statement = "In oder to demonstrate that Mat0-RNA3 can be used as an effective tol in recombination studies, we apply i to examine the recombination activity of too specific sequences derived frm the HCV genome."
    statement = input("Enter your statement below to have spelling check: \n")

    # Get correction results
    results = correct_spell(statement, unigram_fd, bigram_cfd, vocab_size, correct_word)
    
    # Display results
    display_results(statement, results)

if __name__ == "__main__":
    main()
