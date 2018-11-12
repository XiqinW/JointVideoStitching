import cv2
import numpy as np
import math
import resources.libs.Deng.doubly_linked_list as doubly_linked_list

FRAME_SIZE = (720, 1280, 3)
GRID_ = (5, 5)
MIN_FEATURE_NUM = 800


class FeatureDetector:
    FAST_FEATURE_DETECTOR_TYPE = 0

    def __init__(self):
        pass

    def fast(self, threshold=5, on_=True):
        return cv2.FastFeatureDetector_create(threshold, nonmaxSuppression=on_,
                                              type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)

    def my_fast(self, img, threshold=5, circumference=16, nums=9, non_max_suppression=True, n_m_s_window=5):
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # cv2.imshow('img_gray', img_gray)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # this part could be accelerated with Graphic card
        key_points = []
        for x in range(3, img.shape[0] - 3):
            for y in range(3, img.shape[1] - 3):

                diff_list = []
                center_value = img_gray[x][y]

                diff_list.append(abs(int(center_value) - int(img_gray[x - 3][y])))
                diff_list.append(abs(int(center_value) - int(img_gray[x][y + 3])))
                diff_list.append(abs(int(center_value) - int(img_gray[x + 3][y])))
                diff_list.append(abs(int(center_value) - int(img_gray[x][y - 3])))

                nums_window_exceed_threshold = 0

                for i in range(4):
                    if diff_list[i] > threshold:
                        nums_window_exceed_threshold += 1

                if nums_window_exceed_threshold > 2:
                    linked_list = doubly_linked_list.DoublyLinkedList()

                    for i in range(4):
                        linked_list.append(diff_list[i])

                    diff_list.append(abs(int(center_value) - int(img_gray[x - 1][y + 3])))
                    diff_list.append(abs(int(center_value) - int(img_gray[x - 2][y + 2])))
                    diff_list.append(abs(int(center_value) - int(img_gray[x - 3][y + 1])))

                    diff_list.append(abs(int(center_value) - int(img_gray[x + 3][y + 1])))
                    diff_list.append(abs(int(center_value) - int(img_gray[x + 2][y + 2])))
                    diff_list.append(abs(int(center_value) - int(img_gray[x + 1][y + 3])))

                    diff_list.append(abs(int(center_value) - int(img_gray[x + 1][y - 3])))
                    diff_list.append(abs(int(center_value) - int(img_gray[x + 2][y - 2])))
                    diff_list.append(abs(int(center_value) - int(img_gray[x + 3][y - 1])))

                    diff_list.append(abs(int(center_value) - int(img_gray[x - 1][y - 3])))
                    diff_list.append(abs(int(center_value) - int(img_gray[x - 2][y - 2])))
                    diff_list.append(abs(int(center_value) - int(img_gray[x - 3][y - 1])))

                    for i in range(4):
                        for j in range(3):
                            linked_list.insert(i * 4, diff_list[4 + i * 3 + j])
                    counter = 0
                    score = 0
                    node = linked_list.head
                    nums_window_exceed_threshold = 0
                    while counter < circumference + nums:
                        if node.data > threshold:
                            nums_window_exceed_threshold += 1
                        else:
                            nums_window_exceed_threshold = 0
                        node = node.next_
                        counter += 1

                    if nums_window_exceed_threshold > nums - 1:
                        for i in range(16):
                            node = linked_list.get_item(i)
                            score += node.data
                        key_points.append((x, y, score))

                    linked_list.clear()

        if non_max_suppression:
            key_points_nums = len(key_points)
            if key_points_nums > 1:
                while True:
                    counter = 0
                    for i in range(key_points_nums):
                        point = key_points.pop()
                        for remain_point in key_points:
                            if abs(remain_point[0] - point[0]) < n_m_s_window and abs(
                                    remain_point[1] - point[1]) < n_m_s_window and remain_point[2] > point[2]:
                                counter = 0
                                break
                            counter += 1
                        if counter:
                            key_points = [point] + key_points
                    if key_points_nums == len(key_points):
                        break
                    key_points_nums = len(key_points)

        return key_points


def detect_features(path):
    feature_detector = FeatureDetector()
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
            fast = feature_detector.fast()

            mask[int(i * grid_size[0]):int((i + 1) * grid_size[0]),
            int(j * grid_size[1]):int((j + 1) * grid_size[1])] = 1

            grid_kp = fast.detect(frame, mask)

            cv2.drawKeypoints(frame_kp, grid_kp, frame_kp, color=(0, 255, 0))
            kp += grid_kp

    cv2.imshow('frame_kp', frame_kp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
