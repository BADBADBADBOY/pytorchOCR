import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from .DataAgument import transform_image_add,transform_img_shape,transform_image_one
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
#         img = transform_image_add(img)
        img = transform_image_one(img)
        img = transform_img_shape(img,self.config['base']['img_shape'])
        img = Image.fromarray(img)
        img = transforms.ToTensor()(img)
        img.sub_(0.5).div_(0.5)
        label = self.label_list[index]
        return (img,label)
    
    
class CRNNProcessTrainLmdb(Dataset):
    def __init__(self, config):
        self.env = lmdb.open(
            config['trainload']['train_file'],
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode('utf-8')))
            self.nSamples = nSamples

        self.config = config

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key.encode('utf-8'))

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert('RGB')
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]

            label_key = 'label-%09d' % index
            label = txn.get(label_key.encode('utf-8'))
            
        if self.config['base']['is_gray']:
            img = img.convert('L')
        img = np.array(img)
        
        img = transform_image_one(img)
        img = transform_img_shape(img,self.config['base']['img_shape'])
        img = Image.fromarray(img)
        img = transforms.ToTensor()(img)
        img.sub_(0.5).div_(0.5)

        return (img, label) 

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