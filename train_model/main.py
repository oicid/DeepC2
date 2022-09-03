# I need to build a neural networks to transform an image into a vector of length 128.
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import time
from tqdm import tqdm
from LinearNet import LinearNet
from LoadDataset import MyDataset
from contrastive import ContrastiveLoss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model=None):
    if model:
        net = torch.load(model).to(device)
        print("Load model from disk.")
    else:
        net = LinearNet().to(device)
    
    t = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    train_data = MyDataset('train_2260.txt', transform=t)
    test_data = MyDataset('test_2260.txt', transform=t)
    train_loader = DataLoader(train_data, batch_size=64, num_workers=4, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, num_workers=4, shuffle=True)

    lr = 1e-4
    momentum = 0
    threshold = 0.02
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    loss_f = ContrastiveLoss()

    start_time = time.time()
    mark = int(start_time)
    net.train()
    print("Start.")
    for epoch in range(30):
        loss = 0.0
        for data in tqdm(train_loader):
            img1, img2, label = data
            out1, out2 = net(img1.to(device), img2.to(device))
            loss_o = loss_f(out1, out2, label.to(device))
            loss += loss_o.data.item()
            optimizer.zero_grad()
            loss_o.backward()
            optimizer.step()

        right_num_test, label_num_test = test(net, test_loader, threshold, mark)
        print(f'[Epoch {epoch+1}] all: {label_num_test}, true: {right_num_test}, '
              f'rate: {right_num_test / label_num_test}, time: {time.time() - start_time}')
    saved_model = f'models/Siamese_big2_{mark}.pt'
    torch.save(net, saved_model)


def test(net, data_loader, threshold, mark, val=False):
    right_num_all, label_num_all = 0, 0
    net.eval()
    for i, data in enumerate(data_loader):
        img1, img2, label = data
        vec1, vec2 = net(img1.to(device), img2.to(device))
        result = torch.nn.functional.pairwise_distance(vec1, vec2)
        res_comp = torch.eq(torch.sign(torch.ceil(result - threshold)), label.to(device))
        sum_tmp = res_comp.sum().data.item()
        right_num_all += sum_tmp
        label_num_all += res_comp.shape[0]
    return right_num_all, label_num_all


def val(model):
    try:
        net = torch.load(model).to(device)
    except FileNotFoundError as e:
        print("Model does not exist.")
        return

    threshold = 0.02
    t = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    val_data = MyDataset('val_3138.txt', transform=t)
    val_loader = DataLoader(val_data, batch_size=64)
    mark = int(time.time())
    print(f'[VAL] start at {mark}')
    right_num_test, label_num_test = test(net, val_loader, threshold, mark, val=True)
    print(f'[VAL] all: {label_num_test}, true: {right_num_test}, rate: {right_num_test / label_num_test}')


if __name__ == "__main__":
    train()
