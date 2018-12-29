import cv2
import numpy as np
import time
import logging

FRAME_SIZE = (720, 1280, 3)
GRID_ = (5, 5)
MIN_FEATURE_NUM = 800
FAST_THRESHOLD = 50

# G2 in paper BRIEF: Binary Robust Independent Elementary Features⋆ Michael Calonder
mean = [0, 0]
cov = [[33 ** 2 / 25, 0], [0, 33 ** 2 / 25]]  # diagonal covariance
BRIEF_x, BRIEF_y = np.random.multivariate_normal(mean, cov, 1024).T
BRIEF_x = BRIEF_x.astype(np.int)
BRIEF_y = BRIEF_y.astype(np.int)


class MyFast:
    def __init__(self, threshold=5, circumference=16, nums=9, non_max_suppression=True, n_m_s_window=3):
        self.threshold = int(threshold)
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
        # There seems to be a bug here.
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
                            continue
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


def detect_features(im):
    t = time.time()
    feature_detector = FeatureDetector()
    original_frame = im
    frame = original_frame.copy()

    if frame.shape != FRAME_SIZE:
        frame = cv2.resize(frame, (FRAME_SIZE[1], FRAME_SIZE[0]), cv2.INTER_LINEAR)

    # as paper said, " we divide a single frame into 5×5 regular grids
    # and for each grid we use an independent FAST feature detector."
    grid_size = [int(FRAME_SIZE[0] / GRID_[0]), int(FRAME_SIZE[1] / GRID_[1])]

    u_score = FAST_THRESHOLD
    score = [u_score for i in range(GRID_[0] * GRID_[1])]
    threshold = score.copy()

    for k in range(10):
        kp = np.asarray([np.asarray([None, None, None])])
        sum_score = 0
        for i in range(GRID_[0]):

            for j in range(GRID_[1]):
                # if j % (GRID_[1] - 1) > 0 and i % (GRID_[0] - 1) > 0:
                #     continue
                # mask = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
                # fast = feature_detector.fast()
                print(
                    (threshold[i * GRID_[1] + j], score[i * GRID_[1] + j], u_score, score[i * GRID_[1] + j] / u_score),
                    end=" | ")
                threshold[i * GRID_[1] + j] = np.power((score[i * GRID_[1] + j] + 1) / (u_score + 1), 1 / 2) * \
                                              threshold[i * GRID_[1] + j]

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
    index_remove = []
    for i in range(len(kp)):
        bitstring = 0
        if kp[i][0] < 16 or kp[i][1] < 16 or kp[i][0] > FRAME_SIZE[0] - 15 or kp[i][1] > FRAME_SIZE[1] - 15:
            index_remove.append(i)
        else:

            local_Gau_blur = cv2.GaussianBlur(
                frame_gray[(kp[i][0] - 16):(kp[i][0] + 16), (kp[i][1] - 16):(kp[i][1] + 16)], (9, 9), 2)
            for i in range(512):
                if local_Gau_blur[BRIEF_x[2 * i], BRIEF_y[2 * i]] < local_Gau_blur[
                    BRIEF_x[2 * i + 1], BRIEF_y[2 * i + 1]]:
                    bitstring += 2 ** i

        bitstring_list.append(bitstring)
    bitstring_list = np.asarray(bitstring_list)
    bitstring_list = np.delete(bitstring_list, index_remove)
    bitstring_list = np.transpose(bitstring_list)
    kp = np.delete(kp, index_remove, axis=0)
    kp = np.column_stack((kp, bitstring_list))

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
    t = time.time()
    index_list = [[], []]
    s = 0
    # this threshold depends on 1. the distance between object and camera
    # 2. speed of camera's movement 3. speed of object's movement
    threshold = 33
    for point_a in kp_a:
        distances = []
        for point_b in kp_b:

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
    print(time.time() - t)
    # distance_list=index_list[1]
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

    return np.asarray([kp_a[:, 0:2], kp_b[index_list][:, 0:2]])


def feature_match2(kp_a, kp_b):
    t = time.time()
    index_list = [[], []]
    # this threshold depends on 1. the distance between object and camera
    # 2. speed of camera's movement 3. speed of object's movement
    threshold = 33
    c = 0
    l_ = len(kp_a)
    for point_a in kp_a:
        print("a  %d / %d" % (c, l_))
        c += 1
        distances = np.asarray([])
        for point_b in kp_b:
            if hamming_is_in_interval(point_a, point_b):
                hamming_distance = hamming_2(int(point_a[3]), int(point_b[3]))
            else:
                hamming_distance = 512
            distances = np.append(distances, hamming_distance)

        min_distance = distances.min()
        index_matched_list = np.argwhere(distances == min_distance)
        index_matched = index_matched_list[0][0]
        index_list[0].append(index_matched)
    c = 0
    l_ = len(kp_b)
    for point_b in kp_b:
        print("a  %d / %d" % (c, l_))
        c += 1
        distances = np.asarray([])
        for point_a in kp_a:
            if hamming_is_in_interval(point_a, point_b):
                hamming_distance = hamming_2(point_b[3], point_a[3])
            else:
                hamming_distance = 512
            distances = np.append(distances, hamming_distance)

        min_distance = distances.min()
        index_matched_list = np.argwhere(distances == min_distance)
        index_matched = index_matched_list[0][0]
        index_list[1].append(index_matched)

    index_remain = []
    for i in range(len(index_list[0])):
        if i == index_list[1][index_list[0][i]]:
            index_remain.append(i)

    kp_a = kp_a[index_remain]
    index_list[0] = np.asarray(index_list[0])
    index_remain = index_list[0][index_remain]
    kp_b = kp_b[index_remain]

    print("matching: %f" % (time.time() - t))
    return np.hstack((kp_a[:, 0:2], kp_b[:, 0:2]))


def hamming_is_in_interval(point_a, point_b):
    return True
    threshold_top = 33
    threshold_low = 1
    if threshold_low < abs((point_b[0] - point_a[0])) < threshold_top and threshold_low < abs(
            point_b[1] - point_a[1]) < threshold_top:
        return True
    else:
        return False


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

                threshold[i * GRID_[1] + j] = np.power((score[i * GRID_[1] + j] + 1) / (u_score + 1), 1 / 2) * \
                                              threshold[i * GRID_[1] + j]

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


def point_normalize(kp):
    N = kp.shape[0]
    u_mean = sum(kp[:, 0]) / N
    v_mean = sum(kp[:, 1]) / N
    S = np.sqrt(2 * N) / np.sqrt(sum((kp[:, 0] - u_mean) ** 2) + sum((kp[:, 1] - v_mean) ** 2))
    H = np.asarray([[S, 0, -u_mean * S], [0, S, -v_mean * S], [0, 0, 1]])

    kp_normalized = []
    for point in kp:
        point = np.asarray([point[0], point[1], 1])
        point_normalized = np.matmul(H, point)
        kp_normalized.append(point_normalized)

    return H, kp_normalized


def homography_RANSAC(A_points, B_points):
    O_A = A_points
    O_B = B_points
    length_matched_points = len(A_points)

    k = 72
    diff_threshold = 5
    inliers_num_list = [[], [], []]
    for i in range(k):
        inliers_num_list[2].append([])

        inliers_num = 0
        index_4_random = []
        while len(index_4_random) < 4:
            rand_index = np.random.randint(length_matched_points, size=1)
            if rand_index[0] not in index_4_random:
                index_4_random.append(rand_index[0])
        homography = caculate_homography(A_points, B_points, index_4_random)

        for idx in range(0, length_matched_points):
            transfered_point_A = np.dot(homography, np.asarray([[A_points[idx][0]], [A_points[idx][1]], [1]]))
            # Happy new year!
            if abs(transfered_point_A[0][0] - B_points[idx][0]) < diff_threshold and abs(
                    transfered_point_A[1][0] - B_points[idx][1]) < diff_threshold:
                inliers_num += 1
                inliers_num_list[2][i].append(idx)
        inliers_num_list[0].append(inliers_num)
        inliers_num_list[1].append(homography)
    final_inliers = inliers_num_list[2][inliers_num_list[0].index(max(inliers_num_list[0]))]

    A = np.zeros((8, 8))
    R = np.zeros((8, 1))
    # H_normalization_A, A_points = point_normalize(A_points)
    # H_normalization_B, B_points = point_normalize(B_points)

    for i in final_inliers:
        x1 = A_points[i][0]
        y1 = A_points[i][1]
        x2 = B_points[i][0]
        y2 = B_points[i][1]
        k0 = -x1
        k1 = -y1
        k2 = -1
        k3 = x1 * x2
        k4 = x2 * y1
        k5 = x1 * y2
        k6 = y1 * y2

        A[0] += k0 * np.asarray([k0, k1, k2, 0, 0, 0, k3, k4], dtype=float)
        A[1] += k1 * np.asarray([k0, k1, k2, 0, 0, 0, k3, k4], dtype=float)
        A[2] += k2 * np.asarray([k0, k1, k2, 0, 0, 0, k3, k4], dtype=float)
        A[3] += k0 * np.asarray([0, 0, 0, k0, k1, k2, k5, k6], dtype=float)
        A[4] += k1 * np.asarray([0, 0, 0, k0, k1, k2, k5, k6], dtype=float)
        A[5] += k2 * np.asarray([0, 0, 0, k0, k1, k2, k5, k6], dtype=float)

        A[6] += np.asarray([k3 * k0, k3 * k1, k3 * k2, k5 * k0, k5 * k1, k5 * k2, k3 * k3 * k5 * k5, k3 * k4 + k5 * k6],
                           dtype=float)
        A[7] += np.asarray([k4 * k0, k4 * k1, k4 * k2, k6 * k0, k6 * k1, k6 * k2, k4 * k3 + k5 * k6, k4 * k4 + k6 * k6],
                           dtype=float)

        R += np.asarray([[-k0 * x2], [-k1 * x2], [-k2 * x2], [-k0 * y2], [-k1 * y2], [-k2 * y2], [-k3 * x2 - k5 * y2],
                         [-k4 * x2 - k6 * y2]], dtype=float)
        pass

    F = np.append(np.linalg.solve(A, R), 1).reshape((3, 3))
    # homography = np.matmul(np.linalg.inv(H_normalization_B), F)
    # homography = np.matmul(homography, H_normalization_A)
    homography = F

    T_A2B = []
    for point in O_A:
        res = np.matmul(homography, np.asarray([[point[0]], [point[1]], [1]]))
        T_A2B.append([res[0][0], res[1][0]])
    # # test = np.matmul(homography, np.asarray([[O_A[0][0]], [O_A[0][1]], [1]]))

    return final_inliers, homography


def caculate_homography(A_points, B_points, index_list):
    x1 = A_points[index_list[0]][0]
    y1 = A_points[index_list[0]][1]
    x2 = B_points[index_list[0]][0]
    y2 = B_points[index_list[0]][1]
    x3 = A_points[index_list[1]][0]
    y3 = A_points[index_list[1]][1]
    x4 = B_points[index_list[1]][0]
    y4 = B_points[index_list[1]][1]
    x5 = A_points[index_list[2]][0]
    y5 = A_points[index_list[2]][1]
    x6 = B_points[index_list[2]][0]
    y6 = B_points[index_list[2]][1]
    x7 = A_points[index_list[3]][0]
    y7 = A_points[index_list[3]][1]
    x8 = B_points[index_list[3]][0]
    y8 = B_points[index_list[3]][1]

    A = np.zeros((8, 8))
    R = np.asarray([[x2], [y2], [x4], [y4], [x6], [y6], [x8], [y8]])

    A[0] = np.asarray([x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1])[:]
    A[1] = np.asarray([0, 0, 0, x1, y1, 1, -x1 * y2, -y1 * y2])[:]
    A[2] = np.asarray([x3, y3, 1, 0, 0, 0, -x4 * x3, -x4 * y3])[:]
    A[3] = np.asarray([0, 0, 0, x3, y4, 1, -x3 * y4, -y3 * y4])[:]
    A[4] = np.asarray([x5, y6, 1, 0, 0, 0, -x6 * x5, -x6 * y5])[:]
    A[5] = np.asarray([0, 0, 0, x5, y5, 1, -x5 * y6, -y5 * y6])[:]
    A[6] = np.asarray([x7, y8, 1, 0, 0, 0, -x8 * x7, -x8 * y7])[:]
    A[7] = np.asarray([0, 0, 0, x7, y7, 1, -x7 * y8, -y7 * y8])[:]
    H_A2B = np.linalg.solve(A, R)
    H_A2B = np.append(H_A2B, 1).reshape((3, 3))

    return H_A2B


def stitch(im_a, im_b, homography):
    corner = np.asarray([[[0], [0], [1]], [[0], [1280], [1]], [[720], [0], [1]], [[720], [1280], [1]]])
    corner_transformed = []
    for point in corner:
        corner_transformed.append(np.matmul(homography, point))
    corner_transformed = np.asarray(corner_transformed)
    max_height = int(np.ceil(max(corner_transformed[:, 0])))
    max_width = int(np.ceil(max(corner_transformed[:, 1])))

    im_a_transformed = cv2.warpPerspective(im_a, homography, (2 * max_width, 2 * max_height))

    cv2.imshow('im_a_transformed', im_a_transformed)
    cv2.imshow('im_b', im_b)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pass
