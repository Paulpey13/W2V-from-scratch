import re

def trouver_mots_doubles(fichier):
    with open(fichier, 'r', encoding='utf-8') as f:
        lignes = f.readlines()

    for num_ligne, ligne in enumerate(lignes, start=1):
        mots = re.findall(r'\b[a-zA-ZÀ-ÿ]+\b', ligne)
        mots_vus = set()
        for mot in mots:
            if mot.lower() in mots_vus and len(mot)>3:
                print(f'Mot "{mot}" répété à la ligne {num_ligne}')
                break
            mots_vus.add(mot.lower())

trouver_mots_doubles("outputs/outputs_10/analogies_others.txt")