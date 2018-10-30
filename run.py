import cv2
import numpy as np

FRAME_SIZE = (720, 1280, 3)
GRID_ = (5, 5)
FAST_FEATURE_THRESHOLD = [5 for i in range(GRID_[0] * GRID_[1])]
MIN_FEATURE_NUM = 800


def feature_detector(path):
    original_frame = cv2.imread(path)
    frame = original_frame.copy()

    if frame.shape != FRAME_SIZE:
        frame = cv2.resize(frame, (FRAME_SIZE[1], FRAME_SIZE[0]), cv2.INTER_LINEAR)

    # as paper said, " we divide a single frame into 5×5 regular grids
    # and for each grid we use an independent FAST feature detector."
    grid_size = [int(FRAME_SIZE[0] / GRID_[0]), int(FRAME_SIZE[1] / GRID_[1])]
    kp = []
    frame_kp = frame.copy()

    for i in range(GRID_[0]):
        for j in range(GRID_[1]):
            mask = np.zeros(frame.shape, np.uint8)
            threshold = FAST_FEATURE_THRESHOLD[i * GRID_[1] + j]
            fast = cv2.FastFeatureDetector_create(threshold, nonmaxSuppression=True,
                                                  type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)

            mask[int(i * grid_size[0]):int((i + 1) * grid_size[0]),
            int(j * grid_size[1]):int((j + 1) * grid_size[1])] = 1

            grid_kp = fast.detect(frame, mask)

            cv2.drawKeypoints(frame_kp, grid_kp, frame_kp, color=(0, 255, 0))
            kp += grid_kp

    cv2.imshow('frame_kp', frame_kp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    frame_path = "./0001.jpg"
    feature_detector(frame_path)


if __name__ == "__main__":
    main()
