from torch.utils.data import Dataset
import pickle
from PIL import Image
import os


template = ["itap of a {}.",
            "a bad photo of the {}.",
            "a origami {}.",
            "a photo of the large {}.",
            "a {} in a video game.",
            "art of the {}.",
            "a photo of the small {}."]

class LowLightCIFAR10Dataset(Dataset):
    def __init__(self, batch_file, transform=None):
        """
        自定义加载 CIFAR-10 数据集的类
        :param batch_file: CIFAR-10 处理后的 .pkl 文件路径
        :param transform: 数据预处理（如 CLIP 转换）
        """
        # 加载 .pkl 文件
        with open(batch_file, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        
        # 提取图像数据和标签
        self.images = data['data']
        self.labels = data['labels']
        
        # 变换
        self.transform = transform
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 读取图像数据并转换为 [32, 32, 3]
        img = self.images[idx].reshape(3, 32, 32).transpose(1, 2, 0)
        
        # 如果定义了变换（如 CLIP_TRANSFORMS），应用变换
        if self.transform:
            img = Image.fromarray(img)  # 将 NumPy 数组转换为 PIL 图像对象
            img = self.transform(img)
        
        # 获取标签
        label = self.labels[idx]
        
        return img, label
    
class LowLightCIFAR10():
    
    dataset_dir = 'Low_Light_CIFAR_10/test_batch_lowlight_'
    
    def __init__(self, root, degree, preprocess):

        self.dataset_dir = os.path.join(root, self.dataset_dir + degree)

        test_preprocess = preprocess
        
        self.test = LowLightCIFAR10Dataset(self.dataset_dir, transform=test_preprocess)
        
        self.template = template
        self.classnames = self.test.classes

