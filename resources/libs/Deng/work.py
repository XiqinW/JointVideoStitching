import cv2
import numpy as np
import math
import resources.libs.Deng.doubly_linked_list as doubly_linked_list

FRAME_SIZE = (720, 1280, 3)
GRID_ = (5, 5)
MIN_FEATURE_NUM = 800
FAST_THRESHOLD = 4.5


class KeyPoint:
    def __init__(self, angle=-1.0, class_id=-1, octave=0, pt=(0.0, 0.0), response=0.0, size=0.0, score=-1):
        self.angle = angle
        self.class_id = class_id
        self.octave = octave
        self.pt = pt
        self.response = response
        self.size = size
        self.score = score


class MyFast:
    def __init__(self, threshold=5, circumference=16, nums=9, non_max_suppression=True, n_m_s_window=5):
        self.threshold = threshold
        self.circumference = circumference
        self.nums = nums
        self.non_max_suppression = non_max_suppression
        self.n_m_s_window = n_m_s_window
        pass

    def detect(self, img, mask=[]):
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # this part could be accelerated with Graphic card

        if len(mask) == 0:
            mask = [[0, 0], [img_gray.shape[0], img_gray.shape[1]]]
        key_points = []
        for x in range(mask[0][0], mask[1][0]):
            for y in range(mask[0][1], mask[1][1]):

                diff_list = []
                center_value = int(img_gray[x][y])

                diff_list.append(abs(center_value - int(img_gray[x - 3][y])))
                diff_list.append(abs(center_value - int(img_gray[x][y + 3])))
                diff_list.append(abs(center_value - int(img_gray[x + 3][y])))
                diff_list.append(abs(center_value - int(img_gray[x][y - 3])))

                if (
                        (diff_list[0] > self.threshold) + (diff_list[1] > self.threshold) + (
                        diff_list[2] > self.threshold) +
                        (diff_list[3] > self.threshold)) > 2:
                    linked_list = doubly_linked_list.DoublyLinkedList()

                    for i in range(4):
                        linked_list.append(diff_list[i])

                    diff_list.append(abs(center_value - int(img_gray[x - 1][y + 3])))
                    diff_list.append(abs(center_value - int(img_gray[x - 2][y + 2])))
                    diff_list.append(abs(center_value - int(img_gray[x - 3][y + 1])))

                    diff_list.append(abs(center_value - int(img_gray[x + 3][y + 1])))
                    diff_list.append(abs(center_value - int(img_gray[x + 2][y + 2])))
                    diff_list.append(abs(center_value - int(img_gray[x + 1][y + 3])))

                    diff_list.append(abs(center_value - int(img_gray[x + 1][y - 3])))
                    diff_list.append(abs(center_value - int(img_gray[x + 2][y - 2])))
                    diff_list.append(abs(center_value - int(img_gray[x + 3][y - 1])))

                    diff_list.append(abs(center_value - int(img_gray[x - 1][y - 3])))
                    diff_list.append(abs(center_value - int(img_gray[x - 2][y - 2])))
                    diff_list.append(abs(center_value - int(img_gray[x - 3][y - 1])))

                    for i in range(4):
                        for j in range(3):
                            linked_list.insert(i * 4, diff_list[4 + i * 3 + j])
                    counter = 0
                    score = 0
                    node = linked_list.head
                    nums_window_exceed_threshold = 0
                    while counter < self.circumference + self.nums and nums_window_exceed_threshold < self.nums:
                        if node.data > self.threshold:
                            nums_window_exceed_threshold += 1
                        else:
                            nums_window_exceed_threshold = 0
                        node = node.next_
                        counter += 1

                    if nums_window_exceed_threshold >= self.nums:
                        for i in range(16):
                            node = linked_list.get_item(i)
                            score += node.data
                        key_points.append([x, y, score])

                    linked_list.clear()

        if self.non_max_suppression:
            key_points_nums = len(key_points)
            if key_points_nums > 1:
                while True:
                    counter = 0
                    for i in range(key_points_nums):
                        point = key_points.pop()
                        for remain_point in key_points:
                            if abs(remain_point[0] - point[0]) < self.n_m_s_window and abs(
                                    remain_point[1] - point[1]) < self.n_m_s_window and remain_point[2] > point[2]:
                                counter = 0
                                break
                            counter += 1
                        if counter:
                            key_points = [point] + key_points
                    if key_points_nums == len(key_points):
                        break
                    key_points_nums = len(key_points)

        result = [[], [], []]

        for i in range(len(key_points)):
            result[0].append(key_points[i][0])
            result[1].append(key_points[i][1])
            result[2].append(key_points[i][2])
        return result


class FeatureDetector:
    FAST_FEATURE_DETECTOR_TYPE = 0

    def __init__(self):
        pass

    def fast(self, threshold=5, non_max_suppression=True):
        return cv2.FastFeatureDetector_create(threshold, non_max_suppression,
                                              type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)

    def my_fast(self, threshold=5, non_max_suppression=True):
        return MyFast(threshold=threshold, circumference=16, nums=9, non_max_suppression=non_max_suppression,
                      n_m_s_window=5)


def detect_features(path):
    feature_detector = FeatureDetector()
    original_frame = cv2.imread(path)
    frame = original_frame.copy()

    if frame.shape != FRAME_SIZE:
        frame = cv2.resize(frame, (FRAME_SIZE[1], FRAME_SIZE[0]), cv2.INTER_LINEAR)

    # as paper said, " we divide a single frame into 5×5 regular grids
    # and for each grid we use an independent FAST feature detector."
    grid_size = [int(FRAME_SIZE[0] / GRID_[0]), int(FRAME_SIZE[1] / GRID_[1])]

    frame_kp = frame.copy()
    u_score = FAST_THRESHOLD * 2
    score = [u_score for i in range(GRID_[0] * GRID_[1])]
    threshold = score.copy()

    for k in range(2):
        kp = []
        sum_score = 0
        for i in range(GRID_[0]):
            for j in range(GRID_[1]):
                # mask = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
                # fast = feature_detector.fast()
                print(
                    (threshold[i * GRID_[1] + j], score[i * GRID_[1] + j], u_score, score[i * GRID_[1] + j] / u_score),
                    end=" | ")
                threshold[i * GRID_[1] + j] = 1 + math.pow(score[i * GRID_[1] + j] / u_score, 1 / 2) * threshold[
                    i * GRID_[1] + j]

                # threshold[i * GRID_[1] + j] = 1 + score[i * GRID_[1] + j] / u_score * threshold[i * 5 + j]

                print(threshold[i * GRID_[1] + j], end=" | ")
                fast = feature_detector.my_fast(
                    threshold=threshold[i * 5 + j],
                    non_max_suppression=True)

                grid_mask = [[int(i * grid_size[0]), int(j * grid_size[1])],
                             [int((i + 1) * grid_size[0]), int((j + 1) * grid_size[1])]]

                if grid_mask[0][0] == 0:
                    grid_mask[0][0] = 3
                if grid_mask[0][1] == 0:
                    grid_mask[0][1] = 3

                if grid_mask[1][0] == FRAME_SIZE[0]:
                    grid_mask[1][0] = FRAME_SIZE[0] - 3
                if grid_mask[1][1] == FRAME_SIZE[1]:
                    grid_mask[1][1] = FRAME_SIZE[1] - 3

                grid_kp = fast.detect(frame, grid_mask)

                # cv2.drawKeypoints(frame_kp, grid_kp, frame_kp, color=(0, 255, 0))
                # grid_kp = np.asarray(grid_kp)
                grid_score = sum(grid_kp[2])
                sum_score += grid_score
                score[i * GRID_[1] + j] = grid_score
                kp.append(grid_kp)
                print(str(k) + " grid " + str((i, j)) + " got " + str(len(grid_kp[0])) + " features in total")
        u_score = sum_score / (GRID_[0] * GRID_[1])
    print(len(kp[0][0]))
    for grid_key_point in kp:
        for i in range(len(grid_key_point[0])):
            frame_kp = cv2.circle(frame, (grid_key_point[1][i], grid_key_point[0][i]), 3, (0, 255, 0), 1)
    cv2.imshow('frame_kp', frame_kp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
