# NLP Project Aix-Marseille University

This project was made Paul Peyssard and Idir Saidi for the NLP project of the Master 2 Artificial Intelligence and Machine Learning at Aix-Marseille University

# Word2Vec Implementation

## Description

This repository contains a Python implementation of the Word2Vec algorithm using negative sampling and gradient descent. The implementation is focused on the text from "Le Comte de Monte Cristo" by Alexandre Dumas.

The script includes the processes of text tokenization, embedding initialization, loss function calculation, training the model with stochastic gradient descent, and evaluating the word embeddings using cosine similarity.

The second part is about words analogy and how to find those using vector computations.

## Requirements

- Python 3
- NumPy
- Pandas
- Matplotlib
- Regular Expressions (re)
- Collections

## The different functions :

### 1. `tokenize_text(text)`
   - **Purpose**: Tokenizes the input text into words, assigns a unique ID to each word, and returns the tokenized text, a word-to-ID mapping, and an ID-to-word mapping.
   - **Parameters**:
     - `text` (str): The input text to be tokenized.

### 2. `get_distrib(tokenized_text)`
   - **Purpose**: Computes the probability distribution of words in the tokenized text.
   - **Parameters**:
     - `tokenized_text` (list of int): The tokenized text represented as a list of word IDs.

### 3. `initialize_embeddings(text, n, minc)`
   - **Purpose**: Initializes random word embeddings of dimension `n` for the vocabulary in the text.
   - **Parameters**:
     - `text` (str): The input text.
     - `n` (int): The dimensionality of the word embeddings.
     - `minc` (int): A threshold parameter for word occurrences.

### 4. `sigmoid(x)`
   - **Purpose**: Computes the sigmoid function of `x`.
   - **Parameters**:
     - `x` (float or np.ndarray): The input value or array.

### 5. `negative_sampling_loss(positive_embedding, negative_embeddings)`
   - **Purpose**: Computes the loss using negative sampling for word embeddings.
   - **Parameters**:
     - `positive_embedding` (np.ndarray): The word embedding of the target word.
     - `negative_embeddings` (np.ndarray): The word embeddings of negative samples.

### 6. `plot_loss(loss_values)`
   - **Purpose**: Plots the loss values over epochs and saves the plot as an image file.
   - **Parameters**:
     - `loss_values` (list of float): The loss values to be plotted.

### 7. `w2v(text, n, L, k, eta, e, minc)`
   - **Purpose**: Implements the Word2Vec algorithm from scratch to learn word embeddings from text.
   - **Parameters**:
     - `text` (str): The input text.
     - `n` (int): The dimensionality of the word embeddings.
     - `L` (int): The context window size.
     - `k` (int): The number of negative samples.
     - `eta` (float): The learning rate.
     - `e` (int): The number of epochs.
     - `minc` (int): A threshold parameter for word occurrences.

### 8. `load_embeddings(embeddings_file)`
   - **Purpose**: Loads word embeddings from a file.
   - **Parameters**:
     - `embeddings_file` (str): The path to the file containing word embeddings.

### 9. `evaluate_similarity(embeddings, evaluation_file)`
   - **Purpose**: Evaluates the similarity between word embeddings using an evaluation file.
   - **Parameters**:
     - `embeddings` (dict): A dictionary mapping words to their embeddings.
     - `evaluation_file` (str): The path to the file used for evaluation.

### 10. `find_missing_words(analogies_text, model_text)`
   - **Purpose**: Identifies words present in the analogies text but missing in the model text.
   - **Parameters**:
      - `analogies_text` (str): Text containing word analogies.
      - `model_text` (str): Text from the word embeddings model.

### 11. `euclidean_distance(vec1, vec2)`
   - **Purpose**: Computes the Euclidean distance between two vectors.
   - **Parameters**:
      - `vec1` (np.ndarray): The first vector.
      - `vec2` (np.ndarray): The second vector.

### 12. `find_analogy(analogies, embeddings)`
   - **Purpose**: Finds the closest word to complete a word analogy from word embeddings.
   - **Parameters**:
      - `analogies` (list of str): A list of word analogies.
      - `embeddings` (dict): A dictionary mapping words to their embeddings.

### 13. `find_analogy_10(analogies, embeddings)`
   - **Purpose**: Finds the 10 closest words to complete a word analogy from word embeddings.
   - **Parameters**:
      - `analogies` (list of str): A list of word analogies.
      - `embeddings` (dict): A dictionary mapping words to their embeddings.

---
