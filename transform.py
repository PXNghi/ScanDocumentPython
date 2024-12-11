from scipy.spatial import distance as dist
import numpy as np
import cv2

def order_points(pts):
    # sắp xếp các điểm dựa trên trục x
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # lấy điểm trái nhất và phải nhất trên trục x
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # sắp xếp điểm trái nhất dựa trên tọa độ trục y để lấy được điểm trái trên (top-left) và trái dưới (bottom-left)
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # khi có được điểm trái trên, tính khoảng cách Euclide giữa trái trên (top-left) và
    # điểm phải nhất theo định lý Pytago, điểm có khoảng cách lớn nhất sẽ là
    # điểm phải dưới (bottom-right)
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    # trả về một mảng chứa các điểm đã sắp xếp theo thứ tự top-left, top-right, bottom-right, bottom-left
    return np.array([tl, tr, br, bl], dtype = "float32")

def four_point_transform(image, pts):
    # lấy mảng vừa sắp xếp và đưa chúng ra thành từng biến riêng lẻ
    rect = order_points(pts)
    print(rect)
    (tl, tr, br, bl) = rect

    # tính chiều rộng của hình bằng cách tính khoảng cách giữa phải dưới (bottom-right) và trái dưới (bottom-left)
    # hoặc khoảng cách giữa điểm phải trên (top-right) và điểm trái trên (top-left)
    # sau đó so sánh hai khoảng cách và lấy khoảng cách xa nhất
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # chiều cao cũng tương tự bằng cách so sánh khoảng cách xa nhất giữa phải trên (top-right) và phải dưới (bottom-right)
    # hoặc trái trên (top-left) và trái dưới (bottom-left)
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))


    # sắp xếp thành một array với các điểm đã tìm được ở trên để tạo thành một hình chữ nhật
    # và tiến hành biến đổi hình ảnh
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # biến đổi hình ảnh
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped