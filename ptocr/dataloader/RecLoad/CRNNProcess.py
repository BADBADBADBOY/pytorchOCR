import lmdb
import torch
import six,re,glob,os
import numpy as np
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from ptocr.dataloader.RecLoad.DataAgument import transform_img_shape,DataAugment
from ptocr.utils.util_function import create_module,PILImageToCV,CVImageToPIL


def get_img(path,is_gray = False):
    img = Image.open(path).convert('RGB')
    if(is_gray):
        img = img.convert('L')
    return img

class CRNNProcessLmdbLoad(Dataset):
    def __init__(self, config,lmdb_type):
        self.config = config
        self.lmdb_type = lmdb_type

        if lmdb_type=='train':
            lmdb_file = config['trainload']['train_file']
            workers = config['trainload']['num_workers']
        elif lmdb_type == 'val':
            lmdb_file = config['valload']['val_file']
            workers = config['valload']['num_workers']
        else:
            assert 1 == 1
            raise('lmdb_type error !!!')

        self.env = lmdb.open(lmdb_file,max_readers=workers,readonly=True,lock=False,readahead=False,meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (lmdb_file))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode('utf-8')))
            self.nSamples = nSamples

        self.transform_label = create_module(config['label_transform']['label_function'])

        self.bg_img = []
        for path in glob.glob(os.path.join(config['trainload']['bg_path'], '*')):
            self.bg_img.append(path)

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
            label = txn.get(label_key.encode('utf-8')).decode()
        label = self.transform_label(label, char_type=self.config['label_transform']['char_type'],t_type=self.config['label_transform']['t_type'])
        if self.config['base']['is_gray']:
            img = img.convert('L')
        img = PILImageToCV(img,self.config['base']['is_gray'])
        if self.lmdb_type == 'train':
            try:
                bg_index = np.random.randint(0, len(self.bg_img))
                bg_img = PILImageToCV(get_img(self.bg_img[bg_index]),self.config['base']['is_gray'])
                img = transform_img_shape(img, self.config['base']['img_shape'])
                img = DataAugment(img, bg_img, self.config['base']['img_shape'])
                img = transform_img_shape(img, self.config['base']['img_shape'])
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]
        elif self.lmdb_type == 'val':
            img = transform_img_shape(img, self.config['base']['img_shape'])
        img = CVImageToPIL(img,self.config['base']['is_gray'])
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
    
    