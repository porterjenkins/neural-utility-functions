import pickle
from find_similar import Item, by_similarity, by_euclidean_distance




with open("cat_vectors_512_grocery.pt", 'rb') as f:
    grocery = pickle.load(f)

del grocery["na"]

items = {}
i = 0
for k, v in grocery.items():
    items[k] = Item(asin='',
                    idx=i,
                    title=k,
                    vector=v)

    i += 14

item_1 = items["Dairy Milk"]
item_2 = items["Cereal"]

diff = item_1.vector - item_2.vector

item_3 = items["Panckakes and Waffles"]
search = item_3.vector + diff

sim = by_euclidean_distance(items = list(items.values()),start = search)

for score, item in sim[:7]:

    print("Item: {}, score: {:.4f}".format(item.title, score))