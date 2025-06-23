import os
from torch.utils.data import Dataset
from .utils import DatasetBase
from .oxford_pets import OxfordPets
import pickle
import torch
from PIL import Image

fmow_classes = [
            "aerial view of an airport",
            "aerial view of an airport hangar",
            "aerial view of an airport terminal",
            "aerial view of an amusement park",
            "aerial view of aquaculture facilities",
            "aerial view of an archaeological site",
            "aerial view of a barn",
            "aerial view of a border checkpoint",
            "aerial view of a burial site",
            "aerial view of a car dealership",
            "aerial view of a construction site",
            "aerial view of a crop field",
            "aerial view of a dam",
            "aerial view of debris or rubble",
            "aerial view of an educational institution",
            "aerial view of an electric substation",
            "aerial view of a factory or power plant",
            "aerial view of a fire station",
            "aerial view of a flooded road",
            "aerial view of a fountain",
            "aerial view of a gas station",
            "aerial view of a golf course",
            "aerial view of a ground transportation station",
            "aerial view of a helipad",
            "aerial view of a hospital",
            "aerial view of an impoverished settlement",
            "aerial view of an interchange",
            "aerial view of a lake or pond",
            "aerial view of a lighthouse",
            "aerial view of a military facility",
            "aerial view of a multi-unit residential area",
            "aerial view of a nuclear power plant",
            "aerial view of an office building",
            "aerial view of an oil or gas facility",
            "aerial view of a park",
            "aerial view of a parking lot or garage",
            "aerial view of a place of worship",
            "aerial view of a police station",
            "aerial view of a port",
            "aerial view of a prison",
            "aerial view of a race track",
            "aerial view of a railway bridge",
            "aerial view of a recreational facility",
            "aerial view of a road bridge",
            "aerial view of a runway",
            "aerial view of a shipyard",
            "aerial view of a shopping mall",
            "aerial view of a single-unit residential area",
            "aerial view of a smokestack",
            "aerial view of a solar farm",
            "aerial view of a space facility",
            "aerial view of a stadium",
            "aerial view of a storage tank",
            "aerial view of a surface mine",
            "aerial view of a swimming pool",
            "aerial view of a toll booth",
            "aerial view of a tower",
            "aerial view of a tunnel opening",
            "aerial view of a waste disposal site",
            "aerial view of a water treatment facility",
            "aerial view of a wind farm",
            "aerial view of a zoo"
        ]

template = ['a centered satellite photo of {}.']


class FMoWDataset(Dataset):
    def __init__(self, env=0, mode=2, data_dir='./data', transform=None):
        """
        Args:
            env (int): The environment index for selecting a specific subset of the dataset.
            mode (int): The mode index for selecting a specific subset of the dataset.
            data_dir (string): Directory with all the data files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        # self.datasets = pickle.load(open(os.path.join(data_dir, 'fmow.pkl'), 'rb'))
        self.data_dir = data_dir
        self.root = os.path.join(data_dir, 'fmow_v1.1')
        self.datasets = pickle.load(open(os.path.join(self.root, 'fmow.pkl'), 'rb'))
        
        self.env = env
        self.mode = mode
        self.transform = transform

    def __len__(self):
        """Returns the total number of samples in the selected subset."""
        return len(self.datasets[self.env][self.mode]['labels'])

    def __getitem__(self, idx):
        """
        Args:
            idx (int or tensor): Index of the sample to be fetched.
        
        Returns:
            tuple: (image_tensor, label_tensor) where image_tensor is the transformed image and label_tensor is the label.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        label_tensor = torch.LongTensor([self.datasets[self.env][self.mode]['labels'][idx]])
        image_tensor = self.transform(self.get_input(idx))
        return image_tensor, label_tensor

    def get_input(self, idx):
        """
        Args:
            idx (int): Index of the image to be fetched.
        
        Returns:
            PIL.Image: The image corresponding to the given index.
        """
        idx = self.datasets[self.env][self.mode]['image_idxs'][idx]
        img_path = os.path.join(self.root, 'images', f'rgb_img_{idx}.png')
        img = Image.open(img_path).convert('RGB')
        return img


class FMoW():

    dataset_dir = 'fmow_v1.1'

    def __init__(self, root, env, preprocess):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        test_preprocess = preprocess
        # self.image_dir = os.path.join(self.dataset_dir, '2750')
        # self.split_path = os.path.join(self.dataset_dir, 'split_zhou_EuroSAT.json')
        self.test = FMoWDataset(env=env, data_dir=root, transform=test_preprocess)
        
        self.template = template
        self.classnames = fmow_classes