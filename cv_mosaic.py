"""将样本库的图像填充到目标图像
\----resources - 存放充当马赛克像素点的图片
   ∟pretreatment - 存放预处理后的样本图
   ∟run
      ∟result - 最终图像输出位置
      ∟target - 需要做马赛克处理的图片
"""
import os
import cv2
import numpy as np
import time
from functools import reduce
import threading


# 文件重命名
def rename():
    rootpath = r'resources'
    files = os.listdir(rootpath)
    filetype = '.jpg'

    index_name = 0
    for file in files:
        oldname = os.path.join(rootpath, file)

        num_bit = 1
        index = index_name
        while(index // 10 != 0):
            num_bit += 1
            index = index // 10

        newname = os.path.join(rootpath, str(index_name).zfill(6)) + filetype

        os.rename(oldname, newname)
        print(newname)

        index_name += 1


# rename()


# 样本图类 - 封装样本图色与样本地址
class Color:
    def __init__(self, path, rgb):
        self.path = path  # 资源路径
        self.rgb = rgb  # 颜色

    def distance(self, rgb):  # 欧几里得距离
        return (rgb[0] - self.rgb[0]) ** 2 + (rgb[1] - self.rgb[1]) ** 2 + (rgb[2] - self.rgb[2]) ** 2


class Mosaic:
    def __init__(self, tar_length=100, tar_width=100, res_length=10, res_width=10):
        self.resPath = "resources"  # 样本图库
        self.files = os.listdir(self.resPath)
        self.pretPath = "pretreatment"  # 预处理后样本图库
        self.tarPath = "run/target/target.jpg"  # 处理为mosaic的主图片
        self.relPath = "run/result/result.jpg"  # 输出图像
        self.tar_length = tar_length
        self.tar_width = tar_width
        self.res_length = res_length
        self.res_width = res_width
        self.colors = self.pretreat()

    # 预处理
    def pretreat(self):
        img = cv2.imread(self.tarPath)
        # 目标图片缩放 - 需自定义 - 例如（1920*1080）缩放为（192*108），此时样本图为（10*10）可使得图像处理后依然为（1920*1080）
        img = cv2.resize(img, (self.tar_length, self.tar_width))
        cv2.imwrite(self.tarPath, img)  # 重新写入
        colors = []

        # 样本图批量缩放
        for file in self.files:
            imgPath = self.resPath + "/" + file
            pimgPath = self.pretPath + "/" + file
            img = cv2.imread(imgPath)
            img = cv2.resize(img, (self.res_length, self.res_width))  # resize样本大小 - 可自定义
            cv2.imwrite(self.pretPath + "/" + file, img)

            b = int(np.mean(img[:, :, 0]))  # 像素颜色均值(b,g,r)
            g = int(np.mean(img[:, :, 1]))
            r = int(np.mean(img[:, :, 2]))

            # tb = reduce(list.__add__, image[:, :, 0].tolist())  # 像素颜色最大值
            # b = np.argmax(np.bincount(tb))
            # tg = reduce(list.__add__, image[:, :, 1].tolist())
            # g = np.argmax(np.bincount(tg))
            # tr = reduce(list.__add__, image[:, :, 2].tolist())
            # r = np.argmax(np.bincount(tr))

            colors.append(Color(pimgPath, (b, g, r)))

        return colors

    # 绘图
    def paint(self, rel_img, color, row, col):
        print("drawing : ", row, "-", col)
        opt = min(self.colors, key=lambda colors: colors.distance(color))
        pixel = cv2.imread(opt.path)
        rel_img[row * self.res_length:(row + 1) * self.res_length, col * self.res_width:(col + 1) * self.res_width] = pixel  # 绘制图 - 步长为样本大小

    def run(self):
        img = cv2.imread(self.tarPath)
        shape = np.shape(img)
        print(shape)
        rel_img = np.zeros((10 * shape[0], 10 * shape[1], 3), dtype=np.uint8)  # 初始化新画布

        # 多线程
        pthread_list = []
        for i in range(shape[0]):
            for j in range(shape[1]):
                b = img[i, j, 0]
                g = img[i, j, 1]
                r = img[i, j, 2]

                pthread_list.append(threading.Thread(target=self.paint, args=(rel_img, (b, g, r), i, j,)))

        # 启动多线程
        for item in pthread_list:
            item.start()
        item.join()

        cv2.imwrite(self.relPath, rel_img)


def timing():
    start = time.time()

    tar_length = 192
    tar_width = 108
    mosaic = Mosaic(tar_length, tar_width)
    # mosaic.pretreat()
    mosaic.run()

    end = time.time()
    s = end - start
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    print("耗时:  %d:%02d:%02d" % (h, m, s))
    print("END")


if __name__ == "__main__":
    timing()
