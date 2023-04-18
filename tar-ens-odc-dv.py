import torch
import numpy as np
import torchvision.transforms as transforms
from loader import ImageNet, save_image
from torch.utils.data import DataLoader
import pretrainedmodels
import Normalize
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def DI(X_in):
    rnd = np.random.randint(299, 330, size=1)[0]
    h_rem = 331 - rnd
    w_rem = 331 - rnd
    pad_top = np.random.randint(0, h_rem, size=1)[0]
    pad_bottom = h_rem - pad_top - 1
    pad_left = np.random.randint(0, w_rem, size=1)[0]
    pad_right = w_rem - pad_left - 1

    c = np.random.rand(1)
    if c <= 1:
        X_out = torch.nn.functional.pad(torch.nn.functional.interpolate(X_in, size=(rnd, rnd)),
                                        (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        return X_out
    else:
        return X_in


device = torch.device('cuda:1' if torch.cuda.is_available() else "cpu")
output_dir = './Outputs_ODCI_v3/'

input_dir = './dataset/images'
input_csv = './dataset/images.csv'
data_transform = transforms.Compose([transforms.Resize(299), transforms.ToTensor()])
X = ImageNet(input_dir, input_csv, data_transform)
batch_size = 25
data_loader = DataLoader(X, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)

inceptionv3 = torch.nn.Sequential(pretrainedmodels.inceptionv3(num_classes=1000, pretrained='imagenet').eval().to(device))
resnet50 = torch.nn.Sequential(pretrainedmodels.resnet50(num_classes=1000, pretrained='imagenet').eval().to(device))
vgg16 = torch.nn.Sequential(Normalize.Interpolate(torch.Size([224, 224]), 'bilinear'),
                            pretrainedmodels.vgg16(num_classes=1000, pretrained='imagenet').eval().to(device))
densenet121 = torch.nn.Sequential(Normalize.Interpolate(torch.Size([224, 224]), 'bilinear'),
                                  pretrainedmodels.densenet121(num_classes=1000, pretrained='imagenet').eval().to(device))
inceptionv4 = torch.nn.Sequential(pretrainedmodels.inceptionv4(num_classes=1000, pretrained='imagenet').eval().to(device))
inceptionresnetv2 = torch.nn.Sequential(pretrainedmodels.inceptionresnetv2(num_classes=1000, pretrained='imagenet').eval().to(device))
t_model = [inceptionv3, resnet50, vgg16, densenet121, inceptionv4, inceptionresnetv2]

num_iter = 10
epsilon = 12.0 / 255
alpha = epsilon / num_iter
success_count = [0, 0, 0, 0, 0, 0]
ODC_number = 2
ODC_stepsize = 6.0 / 255
m = 5
sample_number = 20

for i, [x, name, y, y2] in enumerate(data_loader):
    x = x.to(device)
    y = y.to(device)
    y2 = y2.to(device)
    x_max = torch.clamp(x + epsilon, 0.0, 1.0)
    x_min = torch.clamp(x - epsilon, 0.0, 1.0)

    onehot = torch.nn.functional.one_hot(y2, num_classes=1000).to(device)
    for _ in range(ODC_number):
        x = x.detach().requires_grad_()
        out1 = inceptionv3(x)
        out2 = resnet50(x)
        out3 = vgg16(x)
        out4 = densenet121(x)
        loss = ((out1 + out2 + out3 + out4) / 4. * onehot).sum()
        loss.backward()
        x = x + ODC_stepsize * x.grad.sign()
        x = torch.clamp(x, x_min, x_max)

    for _ in range(num_iter):
        global_grad = 0
        for _ in range(sample_number):
            x_neighbor = x + (torch.randn(x.shape)).to(device) * 0.1
            x_neighbor = x_neighbor.detach().requires_grad_()
            out1 = inceptionv3(DI(x_neighbor))
            out2 = resnet50(DI(x_neighbor))
            out3 = vgg16(DI(x_neighbor))
            out4 = densenet121(DI(x_neighbor))
            logits = (out1 + out2 + out3 + out4) / 4.
            loss = torch.nn.functional.cross_entropy(logits, y2)
            loss.backward()
            global_grad += x_neighbor.grad
        new_grad = global_grad / (1. * sample_number)
        x = x - alpha * torch.sign(new_grad)
        x = torch.clamp(x, x_min, x_max)

    # adv_img_np = x.detach().cpu().numpy()
    # adv_img_np = np.transpose(adv_img_np, (0, 2, 3, 1)) * 255
    # save_image(adv_img_np, name, output_dir)

    for r in range(len(t_model)):
        success_count[r] += (t_model[r](x).argmax(1) == y2).detach().sum().cpu()
    print('%4d :' % ((i + 1) * batch_size), [t * 1.0 / ((i + 1) * batch_size) for t in success_count])
