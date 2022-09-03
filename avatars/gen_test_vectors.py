# Generate vectors from sample images and model
import torch
from PIL import Image
import os
import json
from torchvision import transforms

transformer = transforms.Compose([transforms.Resize((128, 128)),
                                  transforms.ToTensor()])


def convert_images(path):
    images = os.listdir(path)
    imgs = list()
    for image in images:
        img = Image.open(os.path.join(path, image))
        assert len(img.split()) == 3
        t = transformer(img)
        imgs.append({'name': image, 'tensor': t})
    return imgs


def get_vectors(model):
    net = torch.load(model, map_location=torch.device('cpu'))
    imgs = convert_images('images')
    assert len(imgs) % 2 == 0
    for i in range(0, len(imgs), 2):
        v1, v2 = net(torch.unsqueeze(imgs[i]['tensor'], 0), torch.unsqueeze(imgs[i+1]['tensor'], 0))
        imgs[i]['tensor'] = torch.squeeze(v1).tolist()
        imgs[i+1]['tensor'] = torch.squeeze(v2).tolist()
    return imgs


# for botmaster, vectors with labels
vecs = get_vectors("model.pt")
with open('vectors_bm.dat', 'w') as f:
    f.write(json.dumps(vecs))

# for bots, vectors without labels
vecs = [i['tensor'] for i in vecs]
with open('vectors_bt.dat', 'w') as f:
    f.write(json.dumps(vecs))
