import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
# from dogcat import Net  ##重要，若没有引入这个模型代码，加载模型时会找不到模型
from torchvision import datasets, transforms
from PIL import Image
 
classes = ('cat', 'dog')
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load('/data/rpcv/kevin/code/jupyter/modelcatdog.pt')  # 加载模型
    model = model.to(device)
    model.eval()  # 把模型转为test模式
    img = cv2.imread("/data/rpcv/kevin/dataset/dogs-vs-cats-redux-kernels-edition/test/1.jpg")  # 读取要预测的图片
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    img = trans(img)
    img = img.to(device)
    img = img.unsqueeze(0)  # 图片扩展多一维,因为输入到保存的模型中是4维的[batch_size,通道,长，宽]，而普通图片只有三维，[通道,长，宽]
    output = model(img)
    prob = F.softmax(output, dim=1)  # prob是2个分类的概率
    value, predicted = torch.max(prob, 1) # torch.max 返回最大值和最大值的索引号
    pred_class = classes[predicted.item()]
    print('predicted class is {}, probability is {}%'.format(pred_class, round(value.item(), 6) * 100))
