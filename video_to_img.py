import os
import cv2


# 定义保存图片函数
# image:要保存的图片
# pic_address：图片保存地址
# num: 图片后缀名，用于区分图片，int 类型
def save_image(image, address, num):
    pic_address = address + str(num) + '.jpg'
    cv2.imwrite(pic_address, image)


def video_to_pic(video_path, save_path, frame_rate):
    # 读取视频文件
    videoCapture = cv2.VideoCapture(video_path)
    j = 0
    i = 0
    # 读帧
    success, frame = videoCapture.read()
    while success:
        i = i + 1
        # 每隔固定帧保存一张图片
        if i % frame_rate == 0:
            j = j + 1
            save_image(frame, save_path, j)
            print('图片保存地址：', save_path + str(j) + '.jpg')
        success, frame = videoCapture.read()


def save_img(video, path):
    # 视频文件和图片保存地址
    SAMPLE_VIDEO = video
    SAVE_PATH = path

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # 设置固定帧率
    FRAME_RATE = 20
    video_to_pic(SAMPLE_VIDEO, SAVE_PATH, FRAME_RATE)
