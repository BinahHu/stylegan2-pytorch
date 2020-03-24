from io import BytesIO

import lmdb
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img

def train_transform(resolution=256):
    transform_list = [
        transforms.Resize(size=(resolution * 2, resolution * 2)),
        transforms.RandomCrop(resolution),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

def sample_transform(resolution=256):
    transform_list = [
        transforms.Resize(size=(resolution, resolution)),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

class ImgDataset(Dataset):
    def __init__(self, root, size=256, type='train'):
        super(ImgDataset, self).__init__()
        self.root = root
        self.paths = list(Path(self.root).glob('*'))
        transform = None
        if type == 'train':
            transform = train_transform(size)
        elif type == 'sample':
            transform = sample_transform(size)
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path))
        img = img.convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'