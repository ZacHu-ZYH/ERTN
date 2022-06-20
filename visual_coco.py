import os

from pycocotools.coco import COCO
from skimage import io
from matplotlib import pyplot as plt

json_file = r'E:\HuZhaoyu\ERTN\input\annotations\instances_train2017.json'
dataset_dir = r'E:\HuZhaoyu\ERTN\input\train2017/'
coco = COCO(json_file)
catIds = coco.getCatIds(catNms=['fg']) # catIds=1 表示人这一类
imgIds = coco.getImgIds(catIds=catIds ) # 图片id，许多值
for i in range(len(imgIds)):
    img = coco.loadImgs(imgIds[i])[0]
    I = io.imread(dataset_dir + img['file_name'])
    plt.axis('off')
    plt.imshow(I) #绘制图像，显示交给plt.show()处理
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    coco.showAnns(anns)
    fig = plt.gcf()
    plt.axis('off')
    fig.set_size_inches(12/4,12/4) #dpi = 300, output = 700*700 pixels
    
    
    ##############
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.show() #显示图像
 
    