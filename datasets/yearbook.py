import torchvision.datasets as datasets

template = ["a photo of the {}."]            

yearbook_classes = ['male student', 'female student']


class Yearbook():
    def __init__(self, root, year, preprocess):
        
        self.test = datasets.ImageFolder(root=root + '/yearbook/yearbook-split/' + str(year) + '/2', transform=preprocess)
        self.template = template
        self.classnames = yearbook_classes