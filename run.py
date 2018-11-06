import cv2
import resources.libs.Wang.Stitcher as Sticher

if __name__ == "__main__":
    stitcher = Sticher.Stitcher()
    imageA = cv2.imread("./resources/images/0001.jpg")
    imageB = cv2.imread("./resources/images/0002.jpg")

    stitcher.stitch((imageB, imageA))
