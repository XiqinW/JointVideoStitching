

import os
import cv2
from PIL import Image


def image_to_video(image_path, media_path):
    '''
    图片合成视频函数
    :param image_path: 图片路径
    :param media_path: 合成视频保存路径
    :return:
    '''
    # 获取图片路径下面的所有图片名称
    image_names = os.listdir(image_path)
    # 对提取到的图片名称进行排序
    image_names.sort(key=lambda n: int(n[:-4]))
    # 设置写入格式
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    # 设置每秒帧数
    fps = 2  # 由于图片数目较少，这里设置的帧数比较低
    # 读取第一个图片获取大小尺寸，因为需要转换成视频的图片大小尺寸是一样的
    image = Image.open(r'G:\JointVideoStitching\dmoutput\1.jpg')
    # image = Image.open(image_path + image_names[0])
    # im_a = cv2.imread(img_path_a[i])
    # 初始化媒体写入对象
    media_writer = cv2.VideoWriter(media_path, fourcc, fps, image.size)
    # 遍历图片，将每张图片加入视频当中
    for image_name in image_names:
        im = cv2.imread(os.path.join(image_path, image_name))
        media_writer.write(im)
        print(image_name, '合并完成！')
    # 释放媒体写入对象
    media_writer.release()
    print('无声视频写入完成！')

# 图片路径
image_path =  "./dmoutput"
# 视频保存路径+名称
media_path = "video1.mp4"
# 调用函数，生成视频
image_to_video(image_path, media_path)
