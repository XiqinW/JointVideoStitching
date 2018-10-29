import cv2
import numpy as np

FRAME_SIZE = (720, 1280, 3)


def feature_detector(path):
    frame = cv2.imread(path)

    if frame.shape != FRAME_SIZE:
        frame = cv2.resize(frame, FRAME_SIZE, cv2.INTER_LINEAR)

    fast = cv2.FastFeatureDetector_create(threshold=50, nonmaxSuppression=True,
                                          type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
    # as paper said, " we divide a single frame into 5×5 regular grids
    # and for each grid we use an independent FAST feature detec-tor."
    grid_size = [int(FRAME_SIZE[0] / 5), int(FRAME_SIZE[1] / 5)]
    kp = []
    frame_kp = frame.copy()
    for i in range(5):
        for j in range(5):
            grid = frame_kp[int(i * grid_size[0]):int((i + 1) * grid_size[0]),
                   int(j * grid_size[1]):int((j + 1) * grid_size[1])]
            grid_kp = fast.detect(grid, None)

            cv2.drawKeypoints(grid, grid_kp, grid, color=(0, 255, 0))

            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            for k in range(0, len(grid_kp)):
                height = grid_kp[k].pt[0] + i * grid_size[0]
                width = grid_kp[k].pt[1] + j * grid_size[1]
                grid_kp[k].pt = (height, width)
                kp.append(grid_kp[k])

    cv2.imshow('frame', frame)
    cv2.imshow('frame_kp', frame_kp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    frame_path = "./0001.jpg"
    feature_detector(frame_path)


if __name__ == "__main__":
    main()
