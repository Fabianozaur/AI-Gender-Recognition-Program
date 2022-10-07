
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from PIL import Image
from matplotlib import image as img
import random

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def createImageList():
    face_images = []
    non_face_images = []
    face_image_classes = []
    non_face_image_classes = []


    for i in range(1, 51):
        data1 = Image.open('photos/males/m{}.jpg'.format(i)).convert('RGB')
        # data2 = img.imread('images/m{}.jpg'.format(i))

        face_images.append(data1)
        face_image_classes.append(1)
    for i in range(1, 51):
        data1 = Image.open('photos/females/f{}.jpg'.format(i)).convert('RGB')
        # data2 = img.imread('images/f{}.jpg'.format(i))

        face_images.append(data1)
        face_image_classes.append(1)
    for i in range(1, 51):
        data1 = Image.open('photos/random/o{}.jpg'.format(i)).convert('RGB')
        # data2 = img.imread('images/o{}.jpg'.format(i))

        non_face_images.append(data1)
        non_face_image_classes.append(0)


    random.shuffle(face_images)
    random.shuffle(non_face_images)

    pil_train = face_images[:75]
    pil_train.extend(non_face_images[:38])
    pil_test = face_images[25:]
    pil_test.extend(non_face_images[12:])

    image_class_train = face_image_classes[:75]
    image_class_train.extend(non_face_image_classes[:38])
    image_class_test = face_image_classes[25:]
    image_class_test.extend(non_face_image_classes[12:])

    return pil_train, image_class_train, pil_test, image_class_test

class ImageClassifierDataset(Dataset):
    def __init__(self, image_list, image_classes):
        self.images = []
        self.labels = []
        self.classes = list(set(image_classes))
        self.class_to_label = {c: i for i, c in enumerate(self.classes)}
        self.image_size = 100
        self.transforms = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        for image, image_class in zip(image_list, image_classes):
            transformed_image = self.transforms(image)
            self.images.append(transformed_image)
            label = self.class_to_label[image_class]
            self.labels.append(label)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.images)

def returnDatasets():
    pilTrain, classesTrain, pilTest, classesTest = createImageList()

    dataset_train = ImageClassifierDataset(pilTrain, classesTrain)
    dataset_test = ImageClassifierDataset(pilTest, classesTest)

    return dataset_train, dataset_test

# print(returnDatasets())
