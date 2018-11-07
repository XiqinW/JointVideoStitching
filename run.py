import glob as gb
import cv2
import resources.libs.Wang.Stitcher as Sticher


# 拼接函数
def stitch(images, ratio=0.75, reprojThresh=4.0, showMatches=False):
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

    # 返回匹配结果
    return result


if __name__ == "__main__":

    img_path = gb.glob("./resources/images/*.jpg")

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
    print("all done.")
