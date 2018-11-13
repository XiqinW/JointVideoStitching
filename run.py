import glob as gb
import cv2
import logging
import time
import resources.libs.Wang.Stitcher as Sticher
import resources.libs.Deng.work as work


# 拼接函数
def stitch(images, ratio=0.75, reprojThresh=4.0, showMatches=False):
    logger = logging.getLogger(__name__)

    stitcher = Sticher.Stitcher()
    # 获取输入图片
    (imageB, imageA) = images
    # 检测A、B图片的SIFT关键特征点，并计算特征描述子
    (kpsA, featuresA) = stitcher.detectAndDescribe(imageA[int(imageA.shape[0] / 8):imageA.shape[0], :])
    (kpsB, featuresB) = stitcher.detectAndDescribe(imageB[int(imageB.shape[0] / 8):imageB.shape[0], :])

    # 匹配两张图片的所有特征点，返回匹配结果
    M = stitcher.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)

    # 如果返回结果为空，没有匹配成功的特征点，退出算法
    if M is None:
        logger.info("Features don't match")
        return None

    # 否则，提取匹配结果
    # H是3x3视角变换矩阵
    (matches, H, status) = M
    # 将图片A进行视角变换，result是变换后图片
    result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + int(imageB.shape[1]), imageB.shape[0]))
    # result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
    # 将图片B传入result图片最左端
    for k in range(imageB.shape[1]):
        if imageB[int(imageB.shape[0] / 2), k][0] == 0:
            imageB = imageB[:, 0:k - 20]
            break

    result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

    # 检测是否需要显示图片匹配
    if showMatches:
        # 生成匹配图片
        vis = stitcher.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
        # 返回结果
        return (result, vis)
    else:
        # 返回匹配结果
        return result


def wang():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    img_path = gb.glob("./resources/images/ice_skating/*.jpg")

    num = len(img_path)
    st = int(num / 2) + 1

    # 读取拼接图片
    imageA = cv2.imread(img_path[0])
    for i in range(num - 1):
        imageB = cv2.imread(img_path[i + 1])

        # 把图片拼接成全景图
        (result, vis) = stitch((imageA, imageB), showMatches=True)
        cv2.imwrite("./output/" + str(i) + ".jpg", result)
        imageA = result

    logger.info("All done.")


def deng():
    # im = cv2.imread("./resources/images/building/building0001.jpg")
    work.detect_features("./resources/images/ice_skating/0001.jpg")
    # work.detect_features("./resources/images/building/building0001.jpg")

    # worker = work.FeatureDetector()
    # t = time.time()
    # features = worker.my_fast(im, threshold=50)
    # print(len(features))
    # print(time.time() - t)
    # for point in features:
    #     result_im = cv2.circle(im, (point[1], point[0]), 3, (0, 255, 0), -1)
    #
    # cv2.imshow('result_im_0001', result_im)
    #
    # im = cv2.imread("./resources/images/building/building0002.jpg")
    #
    # worker = work.FeatureDetector()
    # t = time.time()
    # features = worker.my_fast(im, threshold=50)
    # print(len(features))
    # print(time.time() - t)
    # for point in features:
    #     result_im = cv2.circle(im, (point[1], point[0]), 3, (0, 255, 0), -1)
    #
    # cv2.imshow('result_im_0002', result_im)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    deng()
