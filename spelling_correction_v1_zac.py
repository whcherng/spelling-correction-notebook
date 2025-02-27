import re
from nltk import edit_distance, bigrams
from nltk.probability import FreqDist, ConditionalFreqDist

# Load and clean corpus
def load_corpus(file_path):
    with open(file_path, 'r', encoding="utf-8") as f:
        text = f.read().lower()
    # Separate punctuation from words and clean
    text = re.sub(r"([.,!?'])", r" \1 ", text)  # Add spaces around punctuation
    text = re.sub(r"[^a-zA-Z'.,!? ]", " ", text)  # Remove numbers/symbols
    text = re.sub(r"\s+", " ", text)  # Collapse multiple spaces
    return text.split()

# Build language models
def build_models(words):
    unigram_fd = FreqDist(words)
    bigram_cfd = ConditionalFreqDist()
    for prev, curr in bigrams(words):
        bigram_cfd[prev][curr] += 1
    return unigram_fd, bigram_cfd

# Generate candidates with prioritization
def generate_candidates(word, vocabulary, max_distance=2):
    candidates = []
    for candidate in vocabulary:
        dist = edit_distance(word, candidate, transpositions=True)
        if dist <= max_distance:
            candidates.append((candidate, dist))
    # Sort by edit distance first
    return [c[0] for c in sorted(candidates, key=lambda x: x[1])][:100]

# Calculate probability with corpus validation
def calculate_probability(word, prev_word, unigram_fd, bigram_cfd, vocab_size):
    # Check if word exists in corpus
    if word not in unigram_fd:
        return 0.0  # Treat OOV words as invalid
    
    # Unigram probability
    unigram_prob = unigram_fd[word] / unigram_fd.N()
    
    # Bigram probability if context exists
    if prev_word and prev_word in bigram_cfd:
        bigram_prob = (bigram_cfd[prev_word][word] + 1) / (unigram_fd[prev_word] + vocab_size)
    else:
        bigram_prob = unigram_prob
    
    return 0.3 * unigram_prob + 0.7 * bigram_prob

# Improved correction logic
def spell_correct(sentence, unigram_fd, bigram_cfd, vocab_size, vocabulary):
    words = sentence.lower().split()
    results = []
    
    # Use top 10% most frequent words as threshold
    common_words = {word for word, _ in unigram_fd.most_common(int(len(unigram_fd)*0.1))}
    
    for i, word in enumerate(words):
        # Skip punctuation-only tokens
        if re.fullmatch(r"[.,!?']+", word):
            continue
            
        # Get previous word context
        prev_word = words[i-1] if i > 0 and words[i-1] in unigram_fd else None
        
        # Generate candidates if:
        # 1. Word is OOV (not in vocabulary), or
        # 2. Word is rare (not in top 10% frequent words)
        if word not in unigram_fd or word not in common_words:
            candidates = generate_candidates(word, vocabulary)
            if candidates:
                scored = []
                for candidate in candidates:
                    # Skip candidate if same as original
                    if candidate == word:
                        continue
                        
                    edit_dist = edit_distance(word, candidate, transpositions=True)
                    candidate_prob = calculate_probability(candidate, prev_word, unigram_fd, bigram_cfd, vocab_size)
                    error_prob = 0.7 ** edit_dist  # Less aggressive error model
                    score = candidate_prob * error_prob
                    scored.append((candidate, candidate_prob, edit_dist, score))
                
                if scored:
                    # Sort by score and distance
                    top_candidates = sorted(scored, key=lambda x: (-x[3], x[2]))[:5]
                    results.append((word, top_candidates))
    
    return results

# Format and display results
def display_results(sentence, correction_results):
    words = sentence.split()
    output = []
    
    # Create mapping of incorrect words to their corrections
    corrections_map = {res[0]: res[1] for res in correction_results}
    
    for word in words:
        if word.lower() in corrections_map:
            output.append(f"*{word}*")  # Highlight incorrect word
        else:
            output.append(word)
    
    print("\nOriginal Sentence with Errors Highlighted:")
    print(" ".join(output))
    
    print("\nCorrection Suggestions:")
    for incorrect_word, suggestions in correction_results:
        print(f"\nIncorrect word: {incorrect_word}")
        print("Top 5 suggestions:")
        for i, (word, prob, dist, score) in enumerate(suggestions, 1):
            print(f"{i}. {word} (Prob: {prob:.6f}, Dist: {dist}, Score: {score:.6f})")

# Main function
def main():
    # Load corpus and build models
    corpus_file = 'output-COVID19.txt'  # Update with your corpus path
    words = load_corpus(corpus_file)
    unigram_fd, bigram_cfd = build_models(words)
    vocab_size = len(unigram_fd)
    vocabulary = list(unigram_fd.keys())
    
    # Example sentence
    sentence = "In oder to demonstrate that Mat0-RNA3 can be used as an effective tol in recombination studies, we apply i to examine the recombination activity of too specific sequences derived frm the HCV genome."
    
    # Get correction results
    results = spell_correct(sentence, unigram_fd, bigram_cfd, vocab_size, vocabulary)
    
    # Display results
    display_results(sentence, results)

if __name__ == "__main__":
    main()
