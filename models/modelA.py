# encoding=utf-8

import os
import numpy as np

import torch
import torch.nn
import torchvision.models as models
from torch.autograd import Variable
import torch.cuda
import torchvision.transforms as transforms

from PIL import Image

img_to_tensor = transforms.ToTensor()


def make_model():
    resmodel = models.resnet50(pretrained=True)
    resmodel.cuda()  # 将模型从CPU发送到GPU,如果没有GPU则删除该行
    return resmodel


# 分类
def inference(resmodel, imgpath):
    resmodel.eval()  # 必需，否则预测结果是错误的

    img = Image.open(imgpath)
    img = img.resize((224, 224))
    tensor = img_to_tensor(img)

    tensor = tensor.resize_(1, 3, 224, 224)
    tensor = tensor.cuda()  # 将数据发送到GPU，数据和模型在同一个设备上运行

    result = resmodel(Variable(tensor))
    result_npy = result.data.cpu().numpy()  # 将结果传到CPU，并转换为numpy格式
    max_index = np.argmax(result_npy[0])
    print(max_index)

    return max_index


# 特征提取
def extract_feature(resmodel, imgpath):
    resmodel.fc = torch.nn.LeakyReLU(0.1)
    resmodel.eval()

    img = Image.open(imgpath)
    img = img.resize((224, 224))
    tensor = img_to_tensor(img)

    tensor = tensor.resize_(1, 3, 224, 224)
    tensor = tensor.cuda()

    result = resmodel(Variable(tensor))
    result_npy = result.data.cpu().numpy()

    return result_npy[0]


if __name__ == "__main__":
    model = make_model()
    folder_list = os.listdir('valid/')
    folder_list.remove('.DS_Store')
    for folder in folder_list:
        img_list = os.listdir('valid/'+folder)
        for img_name in img_list:
            img_path = 'valid/'+folder+'/'+img_name
            print
            inference(model, img_path)
            print
            # extract_feature(model, img_path)