from PIL import Image
import torch.utils.data as data


class MyDataset(data.Dataset):
    def __init__(self, data_path, transform=None, target_transform=None):
        super(MyDataset, self).__init__()
        fh = open(data_path, 'r')
        self.imgs = list()
        for line in fh:
            line = line.rstrip()
            words = line.split(',')
            self.imgs.append((words[0], words[1], int(words[2])))
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img1, img2, label = self.imgs[index]
        t1 = Image.open(img1)
        t2 = Image.open(img2)
        if self.transform is not None:
            t1 = self.transform(t1)
            t2 = self.transform(t2)
        return t1, t2, label

    def __len__(self):
        return len(self.imgs)
