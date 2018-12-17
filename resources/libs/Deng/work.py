import cv2
import numpy as np
import math
import time
import resources.libs.Deng.doubly_linked_list as doubly_linked_list

FRAME_SIZE = (720, 1280, 3)
GRID_ = (5, 5)
MIN_FEATURE_NUM = 800
FAST_THRESHOLD = 20

# G2 in paper BRIEF: Binary Robust Independent Elementary Features⋆ Michael Calonder
mean = [0, 0]
cov = [[33 ** 2 / 25, 0], [0, 33 ** 2 / 25]]  # diagonal covariance
BRIEF_x, BRIEF_y = np.random.multivariate_normal(mean, cov, 1024).T
BRIEF_x = BRIEF_x.astype(np.int)
BRIEF_y = BRIEF_y.astype(np.int)


class MyFast:
    def __init__(self, threshold=5, circumference=16, nums=9, non_max_suppression=True, n_m_s_window=5):
        self.threshold = threshold
        self.circumference = circumference
        self.nums = nums
        self.non_max_suppression = non_max_suppression
        self.n_m_s_window = n_m_s_window

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
                    diff_list = []

                    diff_list.append(abs(center_value - int(img_gray[x - 1][y + 3])))
                    diff_list.append(abs(center_value - int(img_gray[x - 2][y + 2])))
                    diff_list.append(abs(center_value - int(img_gray[x - 3][y + 1])))
                    diff_list.append(abs(center_value - int(img_gray[x - 3][y])))

                    diff_list.append(abs(center_value - int(img_gray[x + 3][y + 1])))
                    diff_list.append(abs(center_value - int(img_gray[x + 2][y + 2])))
                    diff_list.append(abs(center_value - int(img_gray[x + 1][y + 3])))
                    diff_list.append(abs(center_value - int(img_gray[x][y + 3])))

                    diff_list.append(abs(center_value - int(img_gray[x + 1][y - 3])))
                    diff_list.append(abs(center_value - int(img_gray[x + 2][y - 2])))
                    diff_list.append(abs(center_value - int(img_gray[x + 3][y - 1])))
                    diff_list.append(abs(center_value - int(img_gray[x + 3][y])))

                    diff_list.append(abs(center_value - int(img_gray[x - 1][y - 3])))
                    diff_list.append(abs(center_value - int(img_gray[x - 2][y - 2])))
                    diff_list.append(abs(center_value - int(img_gray[x - 3][y - 1])))
                    diff_list.append(abs(center_value - int(img_gray[x][y - 3])))

                    diff_list.append(abs(center_value - int(img_gray[x - 1][y + 3])))
                    diff_list.append(abs(center_value - int(img_gray[x - 2][y + 2])))
                    diff_list.append(abs(center_value - int(img_gray[x - 3][y + 1])))
                    diff_list.append(abs(center_value - int(img_gray[x - 3][y])))

                    diff_list.append(abs(center_value - int(img_gray[x + 3][y + 1])))
                    diff_list.append(abs(center_value - int(img_gray[x + 2][y + 2])))
                    diff_list.append(abs(center_value - int(img_gray[x + 1][y + 3])))
                    diff_list.append(abs(center_value - int(img_gray[x][y + 3])))

                    nums_window_exceed_threshold = 0
                    for i in diff_list:
                        if i > self.threshold:
                            nums_window_exceed_threshold += 1
                            if nums_window_exceed_threshold > 8:
                                score = sum(diff_list)
                                key_points.append([x, y, int(score)])
                                break
                        else:
                            nums_window_exceed_threshold = 0

        key_points_nums = len(key_points)
        res = []
        if self.non_max_suppression and key_points_nums > 1:
            flag = [1 for i in range(key_points_nums)]

            for i in range(key_points_nums):
                for j in range(1, key_points_nums - i):
                    if flag[i]:
                        if key_points[i + j][0] - key_points[i][0] < self.n_m_s_window and \
                                key_points[i + j][1] - key_points[i][1] < self.n_m_s_window:
                            if key_points[i][2] > key_points[i + j][2]:
                                flag[i + j] = 0
                            else:
                                flag[i] = 0
                        else:
                            break
            for i in range(key_points_nums):
                if flag[i]:
                    res.append(key_points[i])

        return res


class FeatureDetector:
    FAST_FEATURE_DETECTOR_TYPE = 0

    def __init__(self):
        pass

    def fast(self, threshold=5, non_max_suppression=True):
        return cv2.FastFeatureDetector_create(threshold=threshold, nonmaxSuppression=non_max_suppression,
                                              type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)

    def my_fast(self, threshold=5, non_max_suppression=True):
        return MyFast(threshold=threshold, circumference=16, nums=9, non_max_suppression=non_max_suppression,
                      n_m_s_window=5)


def detect_features(path):
    t = time.time()
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

    for k in range(10):
        kp = np.asarray([np.asarray([None, None, None])])
        sum_score = 0
        for i in range(GRID_[0]):
            for j in range(GRID_[1]):
                # mask = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
                # fast = feature_detector.fast()
                print(
                    (threshold[i * GRID_[1] + j], score[i * GRID_[1] + j], u_score, score[i * GRID_[1] + j] / u_score),
                    end=" | ")
                threshold[i * GRID_[1] + j] = math.pow((score[i * GRID_[1] + j] + 1) / (u_score + 1), 1 / 2) * \
                                              threshold[
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
                if len(grid_kp):
                    grid_kp = np.asarray(grid_kp)
                    kp = np.vstack((kp, grid_kp))

                    scores = grid_kp[:, 2]
                    grid_score = sum(scores)
                    sum_score += grid_score
                    score[i * GRID_[1] + j] = grid_score
                print(str(k) + " grid " + str((i, j)) + " got " + str(len(grid_kp)) + " features in total")

        if len(kp) > 800:
            break
        else:
            break
            print("run: grid-based feature detection")
            u_score = sum_score / (GRID_[0] * GRID_[1])
    kp = np.delete(kp, 0, axis=0)

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    bitstring_list = []
    for point in kp:
        bitstring = ''
        if point[0] < 16 or point[1] < 16 or point[0] > FRAME_SIZE[0] - 15 or point[1] > FRAME_SIZE[1] - 15:
            pass
        else:

            local_Gau_blur = cv2.GaussianBlur(
                frame_gray[(point[0] - 16):(point[0] + 16), (point[1] - 16):(point[1] + 16)], (9, 9), 2)
            for i in range(512):
                if local_Gau_blur[BRIEF_x[2 * i], BRIEF_y[2 * i]] < local_Gau_blur[
                    BRIEF_x[2 * i + 1], BRIEF_y[2 * i + 1]]:
                    bitstring += '1'
                else:
                    bitstring += '0'

        bitstring_list.append(bitstring)
        pass
    bitstring_list = np.asarray(bitstring_list)
    bitstring_list = np.transpose(bitstring_list)
    kp = np.column_stack((kp, bitstring_list))
    # for i in range(len(kp[0])):
    #     # frame_kp_gray = cv2.circle(frame, (grid_key_point[1][i], grid_key_point[0][i]), 3, (0, 255, 0), 1)
    #     frame_kp = cv2.circle(frame, (kp[1][i], kp[0][i]), 3, (0, 255, 0), 1)
    # print(time.time() - t)
    # cv2.imshow('frame_kp', frame_kp)

    return kp


def hamming(s1, s2):
    """Calculate the Hamming distance between two bit strings"""
    if len(s1) == len(s2):
        res = sum(c1 != c2 for c1, c2 in zip(s1, s2))
        return res
    else:
        return 512


def hamming_2(s1, s2):
    """Calculate the Hamming distance between two bit strings"""
    hamming_dec = s1 ^ s2
    count = 0
    while hamming_dec:
        hamming_dec &= hamming_dec - 1
        count += 1
    return count


def gaussian_kernel_2d_opencv(kernel_size=3, sigma=0):
    kx = cv2.getGaussianKernel(kernel_size, sigma)
    ky = cv2.getGaussianKernel(kernel_size, sigma)
    return np.multiply(kx, np.transpose(ky))


def feature_match(kp_a, kp_b):
    index_list = [[], []]
    s = 0
    for point_a in kp_a:
        distances = []
        for point_b in kp_b:

            # this threshold depends on 1. the distance between object and camera
            # 2. speed of camera's movement 3. speed of object's movement
            threshold = 33
            if abs((point_a[0] - point_b[0])) < threshold and abs(point_a[1] - point_b[1]) < threshold:
                hamming_distance = hamming_2(int(point_a[3], 2), int(point_b[3], 2))
            else:
                hamming_distance = 512
            distances.append(hamming_distance)

        # print("%d / %d" % (s, len(kp_a)))
        s += 1
        min_distance = min(distances)
        distances = np.asarray(distances)
        index_matched_list = np.argwhere(distances == min_distance)
        min_Euclidean = 2156800
        index_matched = index_matched_list[0][0]
        for i in range(len(index_matched_list)):
            Euclidean_distance = (point_a[0] - kp_b[index_matched_list[i][0]][0]) ** 2 + (
                    point_a[1] - kp_b[index_matched_list[i][0]][1]) ** 2
            if Euclidean_distance < min_Euclidean:
                min_Euclidean = Euclidean_distance
                index_matched = index_matched_list[i][0]

        index_list[0].append(index_matched)
        # Hamming distance list
        index_list[1].append(min_Euclidean)
    index_list = np.asarray(index_list)
    idx = np.argsort(index_list[0])
    index_list[0] = index_list[0][idx]
    index_list[1] = index_list[1][idx]
    kp_a = kp_a[idx]
    t = time.time()
    same_index_list = []
    k = 0
    index_list_len = len(index_list[0])
    while k < index_list_len:

        j = 0
        window = [k - 1]
        while True and k + j < index_list_len:

            if index_list[0][k - 1 + j] == index_list[0][k + j]:
                window.append(k + j)

            else:
                break
            j += 1
        k += j + 1
        same_index_list.append(window)
    index_remove = []
    for same_index in same_index_list:
        if len(same_index) > 1:
            distance_list = index_list[1][same_index]
            # del same_index[distance_list.index(min(distance_list))]
            del same_index[np.argwhere(distance_list == distance_list.min())[0][0]]
            index_remove += same_index
        pass

    distance_list = np.delete(index_list[1], index_remove, axis=0)
    index_list = np.delete(index_list[0], index_remove, axis=0)
    kp_a = np.delete(kp_a, index_remove, axis=0)

    index_remove = []
    for i in range(len(distance_list)):
        if distance_list[i] > 21568:
            index_remove.append(i)

    index_list = np.delete(index_list, index_remove, axis=0)
    kp_a = np.delete(kp_a, index_remove, axis=0)
    print(time.time() - t)
    return index_list, kp_a


def detect_in_grids():
    pass


def detect_features_with_cv(path):
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
    u_score = FAST_THRESHOLD * 2
    score = [u_score for i in range(GRID_[0] * GRID_[1])]
    threshold = score.copy()
    for k in range(2):
        kp = []
        sum_score = 0

        for i in range(GRID_[0]):
            for j in range(GRID_[1]):
                mask = np.zeros(frame.shape, np.uint8)

                mask[int(i * grid_size[0]):int((i + 1) * grid_size[0]),
                int(j * grid_size[1]):int((j + 1) * grid_size[1])] = 1

                print(
                    (threshold[i * GRID_[1] + j], score[i * GRID_[1] + j], u_score, score[i * GRID_[1] + j] / u_score),
                    end=" | ")

                threshold[i * GRID_[1] + j] = math.pow((score[i * GRID_[1] + j] + 1) / (u_score + 1), 1 / 2) * \
                                              threshold[
                                                  i * GRID_[1] + j]

                # threshold[i * GRID_[1] + j] = 1 + score[i * GRID_[1] + j] / u_score * threshold[i * 5 + j]

                print(threshold[i * GRID_[1] + j], end=" | ")

                fast = feature_detector.fast(threshold=int(threshold[i * 5 + j]), non_max_suppression=True)

                grid_kp = fast.detect(frame, mask)

                grid_score = 0
                for point in grid_kp:
                    grid_score += point.response * threshold[i * GRID_[1] + j]
                sum_score += grid_score
                score[i * GRID_[1] + j] = grid_score
                print(str(k) + " grid " + str((i, j)) + " got " + str(len(grid_kp)) + " features")
                kp += grid_kp
        u_score = sum_score / (GRID_[0] * GRID_[1])
    cv2.drawKeypoints(frame_kp, kp, frame_kp, color=(0, 255, 0))
    cv2.imshow('frame_kp', frame_kp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
