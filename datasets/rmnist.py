import torchvision.datasets as datasets


template = ["itap of a {}.",
            "a bad photo of the {}.",
            "a origami {}.",
            "a photo of the large {}.",
            "a {} in a video game.",
            "art of the {}.",
            "a photo of the small {}."]
rmnist_classes = ['number:"0"', 'number:"1"', 'number:"2"', 'number:"3"', 'number:"4"', 'number:"5"', 'number:"6"', 'number:"7"', 'number:"8"', 'number:"9"']



class RMNIST():
    def __init__(self, root, rotation_angle, preprocess):
        
        self.test = datasets.ImageFolder(root=root + '/rmnist/' + str(rotation_angle) + '/2', transform=preprocess)
        self.template = template
        self.classnames = rmnist_classes