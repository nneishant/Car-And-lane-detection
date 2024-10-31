import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 280, 20)
    return canny


def region_of_interest(image):
    height = 550
    polygons = np.array([
        [(450, 610),
         (1000, height),
         (1330, 610)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

    return line_image


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    try:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope < 0:
                left_fit.append((slope, intercept))
                right_fit.append((0.82578829, -80.48243243))  # 147.48243243
            else:

                right_fit.append((slope, intercept))
                left_fit.append((-6.94968944e-01, 9.01574534e+02))


        left_fit_average = np.average(left_fit, axis=0)
        right_fit_average = np.average(right_fit, axis=0)
        left_line = make_coordinates(image, left_fit_average)
        right_line = make_coordinates(image, right_fit_average)

    finally:
        pass
    return np.array([left_line, right_line])


def make_coordinates(image, line_parameters):
    try:
        slope, intercept = line_parameters
        y1 = image.shape[0]
        y2 = int(y1 * (3 / 5))
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
    finally:
        pass
    return np.array([x1, y1, x2, y2])


cap = cv2.VideoCapture('new4.mp4')
while (cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    try:
        lines = cv2.HoughLinesP(cropped_image, 3, np.pi / 180, 40, np.array([]), minLineLength=10, maxLineGap=100)
        averaged_lines = average_slope_intercept(frame, lines)
        line_image = display_lines(frame, averaged_lines)
        combo_image = cv2.addWeighted(frame, .8, line_image, 1, 1)
    finally:
        pass

    cv2.namedWindow("preview - Final Line Detection")
    cv2.imshow("preview - Final Line Detection", combo_image)


    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

cap1 = cv2.VideoCapture('new4.mp4')
car_cascade = cv2.CascadeClassifier('cars.xml')

while True:
    ret, frames = cap.read()
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.1, 9)
    # if str(np.array(cars).shape[0]) == '1':
    #     i += 1
    #     continue
    for (x,y,w,h) in cars:
        plate = frames[y:y + h, x:x + w]
        cv2.rectangle(frames,(x,y),(x +w, y +h) ,(51 ,51,255),2)
        cv2.rectangle(frames, (x, y - 40), (x + w, y), (51,51,255), -2)
        cv2.putText(frames, 'Car', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('car',plate)

    # lab1 = "Car Count: " + str(i)
    # cv2.putText(frames, lab1, (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (147, 20, 255), 3)
    frames = cv2.resize(frames,(600,400))
    cv2.imshow('Car Detection System', frames)
    # cv2.resizeWindow('Car Detection System', 600, 600)
    if cv2.waitKey(50) & 0xFF == ord('q'):

        break
cap.release()
cv2.destroyAllWindows()