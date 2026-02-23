# N-Gram Model: A Statistical Approach to Text Generation

This document explains the code and concepts behind the **N-Gram Model** used in `GenAI_Exp4_NGrams.ipynb`.

---

## 1. The Concept

Before diving into code, it is important to understand that **N-Grams are not "AI"** in the modern sense (like ChatGPT). They are pure **statistics**.

### Imagine a Sliding Window

Think of reading a book through a small window that only shows you a few words at a time.

- If the window shows **"I love eating"**, what word comes next?
- Most likely: "pizza", "apples", "dinner".
- Unlikely: "car", "sky", "jump".

An N-Gram model simply counts how many times "pizza" appears after "I love eating" in a massive library of books. If "pizza" appears 90% of the time, the model predicts "pizza".

### What is "N"?

**N** tells us how big that window is (how many words we look at).

- **N=1 (Unigram)**: We look at **0** previous words. We just pick random words based on how common they are generally.
    - _Result_: "The of and to a..." (Gibberish)
- **N=2 (Bigram)**: We look at **1** previous word to predict the next.
    - _Input_: "New" -> _Prediction_: "York"
- **N=3 (Trigram)**: We look at **2** previous words to predict the next.
    - _Input_: "New York" -> _Prediction_: "City"

### Summary

- **Input**: A sequence of $N-1$ words (the context).
- **Process**: Look up statistics (probabilities) calculated from a training text.
- **Output**: The most likely next word (or a random choice weighted by probability).

---

## 2. Code Walkthrough

This section breaks down the code in the notebook step-by-step.

### Step 1: Data Preparation

We use the `nltk` library (Natural Language Toolkit) to handle the text.

```python
def preprocess_text(text):
    text = text.lower()           # 1. Lowercase: "The" and "the" become the same
    tokens = word_tokenize(text)  # 2. Tokenize: Split text into words/punctuation
    return tokens
```

- **Why?**: Computers need consistency. "Apple" and "apple" should be counted as the same word. Tokenization splits sentences into a list like `['artificial', 'intelligence', 'is', '...']`.

### Step 2: Counting N-Grams

This function creates the "window" snapshots.

```python
def count_ngrams(tokens, n):
    # Create valid indices for the sliding window
    # If tokens=['a','b','c'] and n=2:
    # i=0: tokens[0:2] -> ('a', 'b')
    # i=1: tokens[1:3] -> ('b', 'c')
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    return Counter(ngrams)
```

- **Logic**: It slides a window of size `n` over the list of tokens.
- **Result**: A list of groups (tuples), e.g., `[('artificial', 'intelligence'), ('intelligence', 'is'), ...]`.
- **Counter**: Counts them up. `('artificial', 'intelligence'): 1`.

### Step 3: Training the Model (Probabilities)

Raw counts aren't enough. We need **probabilities**. If "context" appears 100 times, and followed by "A" 20 times, the probability is 20%.

```python
def train_ngram_model(tokens, n):
    ngrams = count_ngrams(tokens, n)
    context_counts = defaultdict(Counter)

    # 1. Group by Context
    for ngram, count in ngrams.items():
        context = ngram[:-1]  # The first N-1 words (History)
        target = ngram[-1]    # The last word (Prediction)
        context_counts[context][target] += count

    # 2. Calculate Probabilities
    model = defaultdict(dict)
    for context, next_words in context_counts.items():
        total_count = sum(next_words.values()) # How many times did this context appear total?
        for word, count in next_words.items():
            model[context][word] = count / total_count # Convert count to %

    return model
```

- **Output**: A dictionary where you look up a context (e.g., "artificial") and get a list of next words with their % chance (e.g., `{'intelligence': 1.0}`).

### Step 4: Generating Text

Now we use the statistics to "write" new text.

```python
def generate_text(model, n, seed_text, max_length=50):
    # ... setup seed ...

    for _ in range(max_length):
        # 1. Look up the current context in our model
        if current_context not in model:
            break

        # 2. Get all possible next words and their probabilities
        possible_words = list(model[current_context].keys())
        probabilities = list(model[current_context].values())

        # 3. Pick one!
        # random.choices picks a word based on its 'weight' (probability)
        # This adds variety. We don't just pick the #1 top word every time.
        next_word = random.choices(possible_words, weights=probabilities, k=1)[0]

        # 4. Append and Slide Window
        result.append(next_word)
        current_context = tuple(result[-(n-1):]) # Update context to include the new word
```

### Summary of Differences

| N-Gram (This Code)                                             | Large Language Model (GPT)                                             |
| :------------------------------------------------------------- | :--------------------------------------------------------------------- |
| **Memory**: Remembers exact word sequences from training data. | **Memory**: Learns abstract "concepts" and patterns in neural weights. |
| **New Text**: Can only combine existing phrases.               | **New Text**: Can create entirely new sentences it never saw before.   |
| **Context**: Limited to `N` words (e.g., 2 or 3).              | **Context**: Can pay attention to thousands of words at once.          |
