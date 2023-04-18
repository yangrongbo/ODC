import os
from PIL import Image
from torch.utils import data
import pandas as pd


def save_image(images, names, output_dir):
    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)
    for i, name in enumerate(names):
        img = Image.fromarray(images[i].astype('uint8'))
        img.save(output_dir + name)

class ImageNet(data.Dataset):
    def __init__(self, dir, csv_path, transforms = None):
        self.dir = dir   
        self.csv = pd.read_csv(csv_path)
        self.transforms = transforms

    def __getitem__(self, index):
        img_obj = self.csv.loc[index]
        filename = img_obj['filename']
        label = img_obj['label']-1
        label2 = img_obj['label2']-1
        img_path = os.path.join(self.dir, filename)
        pil_img = Image.open(img_path).convert('RGB')
        if self.transforms:
            data = self.transforms(pil_img)
        return data, filename, label, label2

    def __len__(self):
        return len(self.csv)
