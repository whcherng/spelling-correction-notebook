import re
import random
from collections import Counter

# Step 1: Text Preprocessing (Cleaning the Corpus)
def clean_text(text):
    """
    Clean the text by removing unnecessary punctuation and converting to lowercase.
    This will help in ensuring a better match when checking for spelling.
    """
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # Remove all non-alphabetical characters
    return text

# Step 2: Damerau-Levenshtein Distance
def damerau_levenshtein(str1, str2):
    """
    Compute the Damerau-Levenshtein distance between two strings.
    It counts the minimum number of single-character edits (insertions, deletions, substitutions, or transpositions) 
    required to change one string into the other.
    """
    len_str1 = len(str1)
    len_str2 = len(str2)
    dp = [[0] * (len_str2 + 1) for _ in range(len_str1 + 1)]
    
    for i in range(len_str1 + 1):
        dp[i][0] = i
    for j in range(len_str2 + 1):
        dp[0][j] = j
    
    for i in range(1, len_str1 + 1):
        for j in range(1, len_str2 + 1):
            cost = 0 if str1[i - 1] == str2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1,  # Deletion
                           dp[i][j - 1] + 1,  # Insertion
                           dp[i - 1][j - 1] + cost)  # Substitution
            
            if i > 1 and j > 1 and str1[i - 1] == str2[j - 2] and str1[i - 2] == str2[j - 1]:
                dp[i][j] = min(dp[i][j], dp[i - 2][j - 2] + cost)  # Transposition
    
    return dp[len_str1][len_str2]

# Step 3: Bigram Model n-gram unigram, bigram, trigram 
def build_bigram_model(text):
    """
    Build a simple bigram model from the corpus. This model will give us the frequency of pairs of consecutive words.
    """
    words = text.split()
    bigrams = list(zip(words[:-1], words[1:]))
    return Counter(bigrams)

def suggest_with_bigram(word, previous_word, bigram_model):
    """
    Suggest corrections for a word using a bigram model.
    The function looks at the previous word and returns suggestions based on the most likely following word.
    """
    suggestions = [bigram[1] for bigram in bigram_model if bigram[0] == previous_word]
    return suggestions if suggestions else [word]  # If no suggestion, return the word itself.

# Step 4: Probabilistic Model (Noisy Channel Model Simplified)
def probabilistic_correction(word, bigram_model, corpus):
    """
    A simplified probabilistic model using word frequency and bigrams.
    The function uses both the frequency of the word and its bigram probability to suggest a correction.
    """
    if word in corpus:
        return [word]  # No correction needed, word is valid in the corpus
    else:
        previous_word = random.choice(corpus.split())  # Take a random previous word as an approximation.
        return suggest_with_bigram(word, previous_word, bigram_model)

# Step 5: Spelling Correction Function (With Top 5 Suggestions)
def correct_spelling(text, corpus):
    """
    The main function to perform spelling correction using Damerau-Levenshtein, Bigram Model, and a simplified 
    Probabilistic model. This function iterates over the text, applying corrections to each word.
    Additionally, it will return top 5 suggestions for each misspelled word based on probabilities.
    """
    # Clean the corpus
    cleaned_corpus = clean_text(corpus)
    
    # Build the bigram model from the cleaned corpus
    bigram_model = build_bigram_model(cleaned_corpus)

    # Split the input text into words and process each word
    words = text.split()
    corrected_text = []
    top_suggestions = {}

    for i, word in enumerate(words):
        # Skip numbers or words that are already correct
        if word.isdigit():
            corrected_text.append(word)
            continue
        if word in cleaned_corpus:
            corrected_text.append(word)  # The word is correct
        else:
            # Apply Damerau-Levenshtein for non-word error correction
            candidate_words = [(corpus_word, damerau_levenshtein(word, corpus_word)) for corpus_word in cleaned_corpus.split()]
            candidate_words.sort(key=lambda x: x[1])  # Sort by edit distance
            best_candidates = [word[0] for word in candidate_words[:5]]  # Top 5 candidates based on distance
            
            # Apply probabilistic correction using bigram model to refine suggestions
            top_5_probabilistic = []
            for candidate in best_candidates:
                suggestions = probabilistic_correction(candidate, bigram_model, cleaned_corpus)
                top_5_probabilistic.extend(suggestions)
            
            top_5_probabilistic = list(set(top_5_probabilistic))  # Remove duplicates
            top_5_probabilistic = sorted(top_5_probabilistic, key=lambda x: bigram_model[(words[i-1], x)], reverse=True)[:5]
            
            top_suggestions[word] = top_5_probabilistic
            corrected_text.append(top_5_probabilistic[0])  # Choose the best suggestion

    return " ".join(corrected_text), top_suggestions

# Step 6: Test the Spelling Correction System
if __name__ == "__main__":
    # Example corpus from output-COVID19.txt (use your own file content here)
    with open("output-COVID19.txt", "r", encoding="utf-8") as file:
        corpus = file.read()
    
    # Sample text with misspellings
    test_text = "As increasing numbers of full-length viral sequenes become available, recombinant or mosaic viruses is being recognise more frequently"
    
    # Perform spelling correction
    corrected_text, top_suggestions = correct_spelling(test_text, corpus)
    
    print("Original Text:", test_text)
    print("Corrected Text:", corrected_text)
    
    print("\nTop 5 Suggestions for Each Misspelled Word:")
    for word, suggestions in top_suggestions.items():
        print(f"Word: {word} => Top 5 Suggestions: {suggestions}")
