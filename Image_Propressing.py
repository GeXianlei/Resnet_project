import os
from PIL import Image
import numpy as np
i = 0
path = './datasets/NonCancerous'
filelist = os.listdir(path)
for item in filelist:
    # print('item name is ',item)
    im = Image.open('./datasets/NonCancerous/'+item)
    im = im.convert("L")
    array = np.array(im)
    b1=(array[256, 1] + array[256, 2] + array[256, 3] + array[256, 4]) / 4
    b2 =(array[257, 1] + array[257, 2] + array[257, 3] + array[257, 4]) / 4
    b3 = (array[255, 1] + array[255, 2] + array[255, 3] + array[255, 4]) / 4
    b4 = (array[258, 1] + array[258, 2] + array[258, 3] + array[258, 4]) / 4
    backgroundcolor = round((b1+b2+b3+b4)/4)
    for col in range(1, 100):
        for row in range(490, 512):
            array[row, col] = backgroundcolor

    for col in range(0, 59):
        for row in range(0, 99):
            array[row, col] = backgroundcolor

    for col in range(0, 512):
        for row in range(0, 20):
            array[row, col] = backgroundcolor

    for col in range(375, 512):
        for row in range(0, 57):
            array[row, col] = backgroundcolor
#    left = 80
#    top = 80
#    right = 420
#    bottom = 420
    i=i+1
    new_img = Image.fromarray(array)
#   new_img.show()
#   im = im.convert("L")  # 转化为黑白图片

    new_img.save(item)
#   im1 = im.crop((left, top, right, bottom))
#   im1.show()
#   im.save(item)
    print('item name is ', item)
print('共转换了',i,'张图片')
