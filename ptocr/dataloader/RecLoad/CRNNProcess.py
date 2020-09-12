import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from .DataAgument import transform_image_add,transform_img_shape
from PIL import Image

def get_img(path,is_gray = False):
    img = Image.open(path).convert('RGB')
    if(is_gray):
        img = img.convert('L')
    return img

class CRNNProcessTrain(Dataset):
    def __init__(self, config):
        super(CRNNProcessTrain,self).__init__()
        with open(config['trainload']['train_file'],'r',encoding='utf-8') as fid:
            self.label_list = []
            self.image_list = []

            for line in fid.readlines():
                line = line.strip('\n').replace('\ufeff','').split('\t') 
                self.label_list.append(line[1])
                self.image_list.append(line[0])

        self.config = config

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        img = get_img(self.image_list[index],is_gray=self.config['base']['is_gray'])
        img = transform_image_add(img)
        img = transform_img_shape(img,self.config['base']['img_shape'])
        img = Image.fromarray(img)
        img = transforms.ToTensor()(img)
        img.sub_(0.5).div_(0.5)
        label = self.label_list[index]
        return (img,label)
    

class alignCollate(object):

    def __init__(self,):
        pass
    def __call__(self, batch):
        images, labels = zip(*batch)
        images = torch.stack(images,0)
        return images,labels
    
class CRNNProcessTest(Dataset):
    def __init__(self, config):
        super(CRNNProcessTest,self).__init__()
        with open(config['testload']['test_file'],'r',encoding='utf-8') as fid:
            self.label_list = []
            self.image_list = []

            for line in fid.readlines():
                line = line.strip('\n').replace('\ufeff','').split('\t') 
                self.label_list.append(line[1])
                self.image_list.append(line[0])

        self.config = config

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        img = get_img(self.image_list[index],is_gray=self.config['base']['is_gray'])
        img = transform_img_shape(img,self.config['base']['img_shape'])
        img = Image.fromarray(img)
        img = transforms.ToTensor()(img)
        img.sub_(0.5).div_(0.5)
        label = self.label_list[index]
        return (img,label)