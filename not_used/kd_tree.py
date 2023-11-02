class Node:
    def __init__(self, point, left=None, right=None):
        self.point = point
        self.left = left
        self.right = right

def build_kd_tree(embeddings_dict, depth=0):
    if not embeddings_dict:
        return None

    k = len(next(iter(embeddings_dict.values()))) 
    print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA :",k)
    axis = depth % k

    sorted_items = sorted(embeddings_dict.items(), key=lambda x: x[1][axis])
    median = len(sorted_items) // 2

    node = Node(sorted_items[median])
    node.left = build_kd_tree(dict(sorted_items[:median]), depth + 1)
    node.right = build_kd_tree(dict(sorted_items[median + 1:]), depth + 1)

    return node



def find_nearest_neighbor(root, target, depth=0):
    if root is None:
        return None

    k = len(target)
    axis = depth % k

    next_branch = None
    opposite_branch = None

    if target[axis] < root.point[1][axis]:
        next_branch = root.left
        opposite_branch = root.right
    else:
        next_branch = root.right
        opposite_branch = root.left

    best = closest_point(root.point, find_nearest_neighbor(next_branch, target, depth + 1), target)

    if distance(target, best) > abs(target[axis] - root.point[1][axis]):
        best = closest_point(root.point, find_nearest_neighbor(opposite_branch, target, depth + 1), target)

    return best


def distance(p1, p2):
    return sum((x - y) ** 2 for x, y in zip(p1, p2))

def closest_point(p1, p2, target):
    if p1 is None:
        return p2
    if p2 is None:
        return p1

    dist1 = distance(p1[1], target)
    dist2 = distance(p2[1], target)

    if dist1 < dist2:
        return p1
    else:
        return p2





def find_analogy(analogies, embeddings):
    results = []
    embeddings_kd_tree = build_kd_tree(embeddings)
    
    for analogy in analogies:
        analogy = analogy.split()
        if len(analogy) != 4:
            continue
            
        word1, word2, word3, expected_word = analogy
        if word1 in embeddings and word2 in embeddings and word3 in embeddings:
            embedding1 = embeddings[word1]
            embedding2 = embeddings[word2]
            embedding3 = embeddings[word3]
            predicted_embedding = embedding1 - embedding2 + embedding3

            closest_word = find_nearest_neighbor(embeddings_kd_tree, predicted_embedding)
            results.append((word1, word2, word3, expected_word, closest_word[0]))

    return results

