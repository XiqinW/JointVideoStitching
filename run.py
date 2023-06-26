import glob as gb
import cv2
import logging
import numpy as np
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
        cv2.imshow('b', imageB)
        # 把图片拼接成全景图
        (result, vis) = stitch((imageA, imageB), showMatches=True)
        cv2.imwrite("./output/" + str(i) + ".jpg", result)
        imageA = result

    logger.info("All done.")


def deng():
    t = time.time()
    im_a = cv2.imread("./video/video_a/1.jpg")
    im_b = cv2.imread("./video/video_b/1.jpg")
    # im_a = cv2.imread("./resources/images/ice_skating/0001.jpg")
    # im_b = cv2.imread("./resources/images/ice_skating/0005.jpg")

    im_a = cv2.resize(im_a, (1280, 720), cv2.INTER_LINEAR)
    im_b = cv2.resize(im_b, (1280, 720), cv2.INTER_LINEAR)
    # work.detect_features("./resources/images/ice_skating/0001.jpg")
    kp = []
    kp.append(work.detect_features(im_a))
    kp.append(work.detect_features(im_b))
    print(time.time() - t)
    # kp.append(work.detect_features("./resources/images/ice_skating/0001.jpg"))
    # kp.append(work.detect_features("./resources/images/ice_skating/0009.jpg"))

    kp_matched = work.feature_match2(kp[0], kp[1])

    A_points = kp_matched[:, 0:2]
    B_points = kp_matched[:, 2:4]

    final_inliers, homography = work.homography_RANSAC(A_points, B_points)

    im = np.hstack((im_a, im_b))

    for i in final_inliers:
        matched = cv2.line(im, (kp_matched[i][1], kp_matched[i][0]),
                           (1280 + kp_matched[i][3], kp_matched[i][2]),
                           (0, 255, 0), 1)
    for i in final_inliers:
        matched = cv2.circle(matched, (kp_matched[i][1], kp_matched[i][0]), 3, (255, 0, 0), 1)
        matched = cv2.circle(matched, (1280 + kp_matched[i][3], kp_matched[i][2]), 3, (255, 0, 0), 1)

        # print("( %d , %d )  ( %d , %d )" % (kp[0][i][1], kp[0][i][0], kp[1][index_list[i]][1], kp[1][index_list[i]][0]))
    print(time.time() - t)
    # 把图片拼接成全景图
    (result, vis) = stitch((im_a, im_b), showMatches=True)
    cv2.imwrite("./output/" + "out.jpg", result)
    cv2.imwrite("./result.jpg", matched)
    cv2.imshow('local_Gau_blur', matched)
    cv2.waitKey(0)
    res = work.stitch(im_a, im_b, homography)
    cv2.imwrite("./output/" + "out.jpg", res)


def duanmu():
    path_a = './resources/images/videoa/'
    path_b = './resources/images/videoa/'
    # path_b = './video/video_b'
    # res_path = './output'
    img_path_a = gb.glob(path_a + "*.jpg")
    img_path_b = gb.glob(path_b + "*.jpg")
    num_a = len(img_path_a)
    num_b = len(img_path_b)
    if num_a != num_b:
        raise Exception
    for i in range(num_a):
        t = time.time()
        im_a = cv2.imread(img_path_a[i])
        im_b = cv2.imread(img_path_b[i])
        # im_a = cv2.imread("./video/video_a/*.jpg")
        # im_b = cv2.imread("./video/video_b/*.jpg")
        # im_a = cv2.imread("./resources/images/ice_skating/0001.jpg")
        # im_b = cv2.imread("./resources/images/ice_skating/0005.jpg")

        im_a = cv2.resize(im_a, (1280, 720), cv2.INTER_LINEAR)
        im_b = cv2.resize(im_b, (1280, 720), cv2.INTER_LINEAR)
        # work.detect_features("./resources/images/ice_skating/0001.jpg")
        kp = []
        kp.append(work.detect_features(im_a))
        kp.append(work.detect_features(im_b))
        print(time.time() - t)
        # kp.append(work.detect_features("./resources/images/ice_skating/0001.jpg"))
        # kp.append(work.detect_features("./resources/images/ice_skating/0009.jpg"))

        kp_matched = work.feature_match2(kp[0], kp[1])

        A_points = kp_matched[:, 0:2]
        B_points = kp_matched[:, 2:4]

        final_inliers, homography = work.homography_RANSAC(A_points, B_points)

        im = np.hstack((im_a, im_b))

        for i in final_inliers:
            matched = cv2.line(im, (kp_matched[i][1], kp_matched[i][0]),
                               (1280 + kp_matched[i][3], kp_matched[i][2]),
                               (0, 255, 0), 1)
        for i in final_inliers:
            matched = cv2.circle(matched, (kp_matched[i][1], kp_matched[i][0]), 3, (255, 0, 0), 1)
            matched = cv2.circle(matched, (1280 + kp_matched[i][3], kp_matched[i][2]), 3, (255, 0, 0), 1)

            # print("( %d , %d )  ( %d , %d )" % (kp[0][i][1], kp[0][i][0], kp[1][index_list[i]][1], kp[1][index_list[i]][0]))
        print(time.time() - t)
        # 把图片拼接成全景图
        (result, vis) = stitch((im_a, im_b), showMatches=True)
        cv2.imwrite("./dmoutput/" + f"out{i}.jpg", result)
        cv2.imwrite("./result.jpg", matched)
        cv2.imshow('local_Gau_blur', matched)
        cv2.waitKey(0)
        # res = work.stitch(im_a, im_b, homography)
        # cv2.imwrite("./dmoutput/" + str(i) + ".jpg",res)




if __name__ == "__main__":
    duanmu()

