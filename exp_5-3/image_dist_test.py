import json
import torch
from PIL import Image
from torchvision import transforms
import os
import time

IMG_FOLDER = 'bots_efficiency_1000'
IMG_FILES = os.listdir(IMG_FOLDER)
MODEL = "model.pt"
THRESHOLD = 0.02
mark = int(time.time())
model = torch.load(MODEL, map_location="cpu")

with open(f'logs_{mark}.csv', 'w', encoding='utf-8') as f:
    f.write('\ufefffilename,time\n')


def get_new_vector():
    vector_file = 'vectors_bm.dat'
    with open(vector_file) as f:
        vecs = json.load(f)
    print(len(vecs))
    for vec in vecs:
        time0 = time.time()
        compare(vec, IMG_FILES)
        time1 = time.time()
        with open(f'logs_{mark}.csv', 'a', encoding='utf-8') as f:
            f.write(f'{vec["name"]},{time1-time0}\n')


def compare(vector, images):
    transformer = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    vec = torch.tensor(vector['tensor'])
    for _ in images:
        image = Image.open(f"{IMG_FOLDER}/{_}")
        img = transformer(image)
        v1, v2 = model(torch.unsqueeze(img, 0), vec)
        distance = torch.pairwise_distance(v1, v2)
        if distance.data.item() < THRESHOLD:
            print(f'[!!!] {vector["name"]} {_} {distance.data.item()}')
            with open(f"{vector['name']}_{_}.txt", "w") as f:
                f.write(str(distance.data.item()))


if __name__ == "__main__":
    get_new_vector()
