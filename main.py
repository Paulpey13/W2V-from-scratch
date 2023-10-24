import pandas as pd
import numpy as np
import math
import random
import re
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import time
# import kd_tree as kd



#################################### -- Read files -- ####################################

with open('Le_comte_de_Monte_Cristo.tok', "r", encoding="utf-8") as file:
    cristo = file.read()

#################################### -- Text tokenization -- ####################################

def tokenize_text(text):
  text=text.lower()
  # text = re.sub(r'[^\w\s]', '', text) #supprime la ponctuation (à enlever si ça gene)

  word_to_id = {}
  id_to_word = {}
  current_id = 0
  tokenized_text = []

  # words = re.findall(r'\w+', text)
  words=text.split(' ')

  for word in words:
      if word not in word_to_id:
          word_to_id[word] = current_id
          id_to_word[current_id] = word
          current_id += 1

      tokenized_text.append(word_to_id[word])

  return tokenized_text, word_to_id, id_to_word


#################################### -- Compute word text probability distribution -- ####################################

def get_distrib(tokenized_text):
    distrib = {}
    total_words = len(tokenized_text)

    for word_id in tokenized_text:
        if word_id in distrib:
            distrib[word_id] += 1
        else:
            distrib[word_id] = 1

    for word_id, count in distrib.items():
        distrib[word_id] = count / total_words

    return distrib

test_text=cristo


#################################### -- Random word embedding -- ####################################

def initialize_embeddings(text, n, minc):
    _, word_to_id, id_to_word = tokenize_text(text)
    vocab_size = len(word_to_id)
    
    word_embeddings = np.random.randn(vocab_size, n)  #Initialiser des embedding aléatoire de dimension n
    return word_embeddings, word_to_id, id_to_word

#################################### -- Loss functions -- ####################################

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def negative_sampling_loss(positive_embedding, negative_embeddings):
    return -np.log(sigmoid(np.dot(positive_embedding, negative_embeddings.T)))

#################################### -- Plot the loss curves-- ####################################

def plot_loss(loss_values):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_values) + 1), loss_values, marker='o', linestyle='-', color='b')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.savefig("loss_curves.png")  #sauvegarder l'image

    # plt.show()

#################################### -- Final w2v function -- ####################################

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

                #Update negative embeddings avec la distribution du vocabulaire
                for _ in range(k):
                    # negative_word_id = np.random.choice(word_ids, p=probabilities)
                    negative_word_id = np.random.choice(vocab_size) #Test de faire juste en random normal pour comparer resultats, à enlever
                    negative_embedding = word_embeddings[negative_word_id]
                    loss = negative_sampling_loss(context_embedding, negative_embedding)
                    gradient = -sigmoid(-np.dot(context_embedding, negative_embedding.T))
                    word_embeddings[negative_word_id] -= eta * gradient * context_embedding

                #Update target and context embeddings
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



params = {
    'text' : cristo, 
    'n': 100,
    'L':  2,
    'k': 10,
    'eta':   0.1,
    'e': 5,
    'minc': 5
}

# w2v(**params) #Commenté car on a deja le fichier d'emebedding

#################################### -- Partie 2 : évaluation -- ####################################

#################################### -- Charger le fichier d'emebedding précedement créer  -- ####################################

# def load_embeddings(embeddings_file):
#     # Charger les embeddings depuis le fichier
#     embeddings = {}
#     with open(embeddings_file, 'r', encoding='ISO-8859-1') as f:
#         next(f)  # Ignorer la première ligne
#         for line in f:
#             parts = line.strip().split(' ')
#             word = parts[0]
#             embedding = np.array([float(x) for x in parts[1:]])
#             embeddings[word] = embedding
#     return embeddings

import numpy as np

def load_embeddings(embeddings_file):
    # Charger les embeddings depuis le fichier
    embeddings = {}
    with open(embeddings_file, 'r', encoding='ISO-8859-1') as f:
        next(f)  # Ignorer la première ligne
        for line in f:
            line = line.strip()
            parts = line.split(' ')
            word = parts[0]
            embedding = np.array([float(x) for x in parts[1:]])
            embeddings[word] = embedding
    return embeddings






def evaluate_similarity(embeddings, evaluation_file):
    correct_count = 0
    total_count = 0

    with open(evaluation_file, 'r', encoding='utf-8') as f:
        for line in f:
            m, m_plus, m_minus = line.strip().split(' ')

            if m in embeddings and m_plus in embeddings and m_minus in embeddings:
                vector_m = embeddings[m]
                vector_m_plus = embeddings[m_plus]
                vector_m_minus = embeddings[m_minus]

                #Compute la similarité cosinus
                similarity_m_m_plus = np.dot(vector_m, vector_m_plus) / (np.linalg.norm(vector_m) * np.linalg.norm(vector_m_plus))
                similarity_m_m_minus = np.dot(vector_m, vector_m_minus) / (np.linalg.norm(vector_m) * np.linalg.norm(vector_m_minus))

                #Vérifier si sim(m, m+) > sim(m, m-)
                if similarity_m_m_plus > similarity_m_m_minus:
                    correct_count += 1

                total_count += 1

    return correct_count / total_count

if __name__ == "__main__":

    embeddings_file = 'word_embeddings.txt'  #Emebedding générés
    evaluation_file = 'Le_comte_de_Monte_Cristo.100.sim'  #Fichier d'évaluation
    nb_experiences = 10  #nb d'expériences à effectuer

    success_rates = []  #Pour stocker les taux de réussite de similarité pour chaque expérience

    #Faire tourner n fois les fonctions afin d'avoir une moyenne et un ecart type des 

    
    # for experiences in range(nb_experiences):
    #     w2v(**params)
    #     embeddings = load_embeddings(embeddings_file)
    #     success_rate = evaluate_similarity(embeddings, evaluation_file)
    #     success_rates.append(success_rate)

    # mean_success_rate = np.mean(success_rates) * 100
    # std_deviation = np.std(success_rates) * 100

    # print(f"Mean Success Rate: {mean_success_rate:.2f}%")
    # print(f"Mean Standard Deviation: {std_deviation:.2f}%")
    #Dernier score obtenu avec 10 experiences : avg mean : 50.4%, avg std : 4.54%
    #Score obtenu par le prof : 53.2% et 3.9%, Etant donné la dimension aléatoire, je pense que c'est compatible

#################################### -- Partie 3 : Pistes à creuser : Analogies -- ####################################


#################################### -- Fonction pour trouver les mots de mes analogies qui ne sont pas dans le corpus -- ####################################

def find_missing_words(analogies_text, model_text):
    model_words = set(re.findall(r'\b\w+\b', model_text))
    
    # Séparer les mots des analogies
    analogies_words = set(re.findall(r'\b\w+\b', analogies_text))
    
    # Trouver les mots qui sont dans "analogies.txt" mais pas dans "word_embeddings.txt"
    missing_words = analogies_words - model_words
    

    return (missing_words)

# missing_words = find_missing_words(analogies, model)
# print(missing_words)
# print(len(missing_words))

import numpy as np

def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)


#################################### -- Premère fonction poru trouver les analogies -- ####################################
#################################### -- (Je vais en faire une autres avec des kd-tree, plus efficace) -- ####################################


def find_analogy(analogies, embeddings):
    results = []
    for analogy in analogies:
        analogy=analogy.split()
        if len(analogy) != 4:
            continue  #Pour éviter les lignes mal formées mais il ne devrait plus avoir de soucis
        word1, word2, word3, expected_word = analogy
        if word1 in embeddings and word2 in embeddings and word3 in embeddings:
            #Calculer du prolongement de mots à deviner
            embedding1 = embeddings[word1]
            embedding2 = embeddings[word2]
            embedding3 = embeddings[word3]
            # predicted_embedding = embedding1 - embedding2 + embedding3
            predicted_embedding = embedding2 - embedding1 + embedding3


            #Trouver le mot le plus proche avec la distance euclidienne
            min_distance = float('inf')
            closest_word = None
            for word, embedding in embeddings.items():
                if word not in [word1, word2, word3]:
                    distance = euclidean_distance(predicted_embedding,embedding)
                    if distance < min_distance:
                        min_distance = distance
                        closest_word = word
            results.append((word1, word2, word3, expected_word, closest_word))

    return results



#Partie ou on analyse l'unique mot le plus proche pour chaque analogie

start_time = time.time()

model=load_embeddings('model.txt')

analogies_files = ['analogies/analogies_conjuguaison.txt', 'analogies/analogies_gender.txt', 'analogies/analogies_localisation.txt', 'analogies/analogies_others.txt']
outputs_files = ['outputs/outputs_unique/analogies_conjuguaison.txt', 'outputs/outputs_unique/analogies_gender.txt', 'outputs/outputs_unique/analogies_localisation.txt', 'outputs/outputs_unique/analogies_others.txt']
outputs_count=-1
for file_path in analogies_files:
    outputs_count=outputs_count+1
    with open(file_path, 'r', encoding='utf-8') as analogies_file:
        analogies = [line.strip() for line in analogies_file.readlines()]
    analogies = [item.lower() for item in analogies]
    results = find_analogy(analogies, model)

    with open(outputs_files[outputs_count], 'w',encoding='utf-8') as file:
        for result in results:
            file.write(f'{result[1]} - {result[0]} + {result[2]} = {result[4]} (Expected: {result[3]})\n')
    
end_time = time.time()
execution_time = end_time - start_time
print(f"Temps d'exécution : {execution_time} secondes")


#Partie ou on analyse les 10 mots les plus proches de l'analogie
def find_analogy_10(analogies, embeddings):
    results = []
    for analogy in analogies:
        analogy = analogy.split()
        if len(analogy) != 4:
            continue  # Pour éviter les lignes mal formées
        word1, word2, word3, expected_word = analogy
        if word1 in embeddings and word2 in embeddings and word3 in embeddings:
            # Calculer du prolongement de mots à deviner
            embedding1 = embeddings[word1]
            embedding2 = embeddings[word2]
            embedding3 = embeddings[word3]
            predicted_embedding = embedding2 - embedding1 + embedding3

            # Trouver les 10 mots les plus proches avec la distance euclidienne
            closest_words = []
            for word, embedding in embeddings.items():
                if word not in [word1, word2, word3]:
                    distance = euclidean_distance(predicted_embedding, embedding)
                    if len(closest_words) < 10 or distance < closest_words[-1][1]:
                        closest_words.append((word, distance))
                        closest_words.sort(key=lambda x: x[1])
                        closest_words = closest_words[:10]
            results.append((word1, word2, word3, expected_word, closest_words))

    return results



start_time = time.time()

# model=load_embeddings('model.txt')

analogies_files = ['analogies/analogies_conjuguaison.txt', 'analogies/analogies_gender.txt', 'analogies/analogies_localisation.txt', 'analogies/analogies_others.txt']
# analogies_files=['analogies/analogies_conjuguaison.txt']
outputs_files = ['outputs/outputs_10/analogies_conjuguaison.txt', 'outputs/outputs_10/analogies_gender.txt', 'outputs/outputs_10/analogies_localisation.txt', 'outputs/outputs_10/analogies_others.txt']
# outputs_files=['outputs/outputs_10/analogies_conjuguaison.txt']
outputs_count=-1
for file_path in analogies_files:
    outputs_count=outputs_count+1
    with open(file_path, 'r', encoding='utf-8') as analogies_file:
        analogies = [line.strip() for line in analogies_file.readlines()]
    analogies = [item.lower() for item in analogies]
    results = find_analogy_10(analogies, model)
    with open(outputs_files[outputs_count], 'w',encoding='utf-8') as file:
        for result in results:
            closest_words_str = ', '.join([f"{word} ({distance:.3f})" for word, distance in result[4]])
            file.write(f'{result[1]} - {result[0]} + {result[2]} = {closest_words_str} (Expected: {result[3]})\n')

    
end_time = time.time()
execution_time = end_time - start_time
print(f"Temps d'exécution : {execution_time} secondes")