import pandas as pd
import numpy as np
import math
import random
import re
from collections import defaultdict, Counter
import matplotlib.pyplot as plt



#################################### -- Compute positif dict of the text -- ####################################

def get_pos_context(text, L):
    word_context_dict = defaultdict(list)
    # text=text.split()
    for i, word in enumerate(text):
        left_bound = max(0, i - L)
        right_bound = min(len(text), i + L + 1)

        context = [text[j] for j in range(left_bound, right_bound) if j != i]

        word_context_dict[word].extend(context)

    word_context_dict = {key: list(set(value)) for key, value in word_context_dict.items()}

    return word_context_dict


#################################### -- Compute negativ dict of text (size = k*len pos) -- ####################################
#Return un dico (aléatoire k*size_pos) de tous les contexte negatif des mots d'un texte

def get_neg_context(tokenized_text, pos_context_dict, k):
    distribution=get_distrib(tokenized_text)
    neg_context_dict = {}

    #Créer un ensemble de tous les tokens uniques dans le texte
    unique_tokens = set(tokenized_text)

    for word, pos_context in pos_context_dict.items():
        #Calculer la taille du nouveau contexte négatif pour chaque mot
        neg_context_size = len(pos_context) * k

        #onvertir unique_tokens en une liste pour pouvoir l'utiliser avec random.sample
        unique_tokens_list = list(unique_tokens)

        #selectionner au hasard des mots parmi les tokens uniques pour former le contexte négatif
        neg_context = random.choices(unique_tokens_list, weights=distribution, k=min(neg_context_size, len(unique_tokens)))
        neg_context_dict[word] = neg_context

    return neg_context_dict

def w2v(text, n, L, k, eta, e, minc):

    word_embeddings, word_to_id, id_to_word = initialize_embeddings(text, n, minc)
    vocab_size, embedding_dim = word_embeddings.shape


    loss_values = []
    for epoch in range(e):
        total_loss = 0
        for i, target_word_id in enumerate(text):
            if i % 1000 == 0:
                print(f"Epoch {epoch+1}/{e}, Iteration {i}/{len(text)}")

            #Sauter les mots qui n'ont pas assez d'occurence
            if target_word_id not in word_to_id:
                continue

            #positive context
            left_bound = max(0, i - L)
            right_bound = min(len(text), i + L + 1)
            context = [text[j] for j in range(left_bound, right_bound) if j != i]

            if not context:
                continue

            for context_word_id in context:
                if context_word_id not in word_to_id:
                    continue

                target_embedding = word_embeddings[word_to_id[target_word_id]]
                context_embedding = word_embeddings[word_to_id[context_word_id]]

                # Update negative embeddings
                for _ in range(k):
                    negative_word_id = np.random.choice(vocab_size)
                    negative_embedding = word_embeddings[negative_word_id]
                    loss = negative_sampling_loss(context_embedding, negative_embedding)
                    gradient = -sigmoid(-np.dot(context_embedding, negative_embedding.T))
                    word_embeddings[negative_word_id] -= eta * gradient * context_embedding

                # Update target and context embeddings
                loss = negative_sampling_loss(context_embedding, target_embedding)
                gradient = -sigmoid(-np.dot(context_embedding, target_embedding.T))
                word_embeddings[word_to_id[target_word_id]] -= eta * gradient * context_embedding
                total_loss += loss

        loss_values.append(total_loss)
        print("Loss function curve saved as loss_curves.png")

        print(f"Epoch {epoch+1}/{e}, Loss: {total_loss}")

    plot_loss(loss_values)
    #Ecriture du fichier d'emebedding
    with open('word_embeddings.txt', 'w', encoding='utf-8') as f:
        f.write(f"{vocab_size} {embedding_dim}\n")
        for i, word_embedding in enumerate(word_embeddings):
            word = id_to_word[i]
            embedding_str = ' '.join([f"{x:.3f}" for x in word_embedding])  #Affiche que 3 chiffres apres la virgule sinon c'est compliqué de lire le fichier
            f.write(f"{word} {embedding_str}\n")
def solve_analogies(embedding_file, analogies):
    # Charger les embeddings
    embeddings = load_embeddings(embedding_file)

    results = []  # Liste pour stocker les résultats

    for analogy in analogies:
        analogy = analogy.strip()
        words = analogy.split(' ')

        if len(words) != 4:
            continue

        word1, word2, word3, word4 = words
        
        # Assurer que les mots sont dans les embeddings
        if word1 not in embeddings or word2 not in embeddings or word3 not in embeddings:
            print(f"Certains des mots de l'analogie ne sont pas dans les embeddings: {word1}, {word2}, {word3}")
            continue
        
        # Récupérer les vecteurs des mots
        vec1 = embeddings[word1]
        vec2 = embeddings[word2]
        vec3 = embeddings[word3]

        # Calculer la différence entre vec2 et vec1, et l'ajouter à vec3
        vec4 = vec1 - vec2 + vec3

        # Filtrer les mots du fichier d'embeddings pour ne conserver que ceux de l'analogie
        # relevant_embeddings = {word: vec for word, vec in embeddings.items() if word in words}

        # Calculer la distance euclidienne entre vec4 et les vecteurs pertinents
        distances = {word: euclidean_distance(vec4, vec) for word, vec in embeddings.items()}

        # Trier les distances par ordre croissant (plus petite distance signifie plus similaire)
        sorted_distances = sorted(distances.items(), key=lambda x: x[1])

        # Le mot le plus similaire est le premier élément de la liste triée
        closest_word = sorted_distances[0][0]

        # Ajouter l'analogie et le mot le plus proche à la liste des résultats
        results.append((analogy, closest_word))

    return results


# Appeler la fonction solve_analogies avec vos données
results = solve_analogies("word_embeddings.txt", analogies)

for analogy, closest_word in results:
    print(f"Analogie : {analogy} -> Mot le plus proche : {closest_word}")