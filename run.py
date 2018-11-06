import glob as gb
import cv2
import resources.libs.Wang.Stitcher as Sticher

if __name__ == "__main__":
    stitcher = Sticher.Stitcher()
    img_path = gb.glob("./resources/images/*.jpg")

    num = len(img_path)
    st = int(num / 2) + 1

    # 读取拼接图片
    # for
    imageA = cv2.imread(img_path[0])
    for i in range(num - 1):
        imageB = cv2.imread(img_path[i + 1])

        # 把图片拼接成全景图
        (result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)
        cv2.imwrite("./output/" + str(i) + ".jpg", result)
        imageA = result
    print("all done.")
