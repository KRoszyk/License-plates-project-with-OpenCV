import numpy as np
import cv2 as cv
from skimage.metrics._structural_similarity import structural_similarity as ssim


def perform_processing(fonts_list, image: np.ndarray) -> str:
    # Main processing function which uses another basic functions :)
    plate_outlines, resized_image = preprocessing(image)
    trans_plate = plate_transform(plate_outlines, resized_image)
    preletters = plate_processing(trans_plate)
    letters = letters_processing(preletters, trans_plate)
    final_result = final_processing(letters, fonts_list)

    return final_result

def preprocessing(image: np.ndarray):
    # This function resizes the original image, converts it to grayscale and looks for outlines to find the license plate in the image
    dim = (640, 480)
    resized = cv.resize(image, dim, interpolation=cv.INTER_AREA)

    # This copy of image will be useful in another function
    plates = resized.copy()

    # We have to convert the image to grayscale, because we need to use it in the bilateral filter
    gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
    b_filter = cv.bilateralFilter(gray, 9, 17, 17)
    # To detect outlines, we use Canny detector
    edged = cv.Canny(b_filter, 30, 250)
    cnts, new = cv.findContours(edged.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    # We sort found contours and then surround those with specified dimensions with a rectangle and draw them on the image
    cnts = sorted(cnts, key=cv.contourArea, reverse=True)[:30]
    for c in cnts:
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.008 * peri, True)
        x, y, w, h = cv.boundingRect(c)
        if w > 200:
            if len(approx) < 50:  # Select the contour with less than 50 corners
                NumberPlateCnt = approx  # This is our approx Number Plate Contour
                # Drawing detected contours with a green colour
                cv.drawContours(resized, [NumberPlateCnt], -1, (0, 255, 0), 3)
            break

    return resized, plates


def plate_transform(resized, plates):
    # Making a perspective transformation for the detected license plate
    resized_copy = resized.copy()
    resized_col = np.shape(resized)[0]
    resized_row = np.shape(resized)[1]

    # Leaving only this part of the photo that has been framed in green, because \
    # in the previous function, we marked the detected license plates with green
    for i in range(0, int(resized_col) - 1):
        for j in range(0, int(resized_row) - 1):
            if resized[i, j, 0] == 0 and resized[i, j, 1] == 255 and resized[i, j, 2] == 0:
                resized_copy[i, j, :] = 255
            else:
                resized_copy[i, j, :] = 0
    # We define the boundaries of the frames once again in order to define their vertex points
    resized_copy_gray = cv.cvtColor(resized_copy, cv.COLOR_BGR2GRAY)
    conts, news = cv.findContours(resized_copy_gray, 1, 2)
    conts = sorted(conts, key=cv.contourArea, reverse=True)[:20]
    x_0, y_0, w_0, h_0 = cv.boundingRect(conts[0])
    area_0 = w_0 * h_0
    rectangled = cv.minAreaRect(conts[0])
    boxi = cv.boxPoints(rectangled)
    boxi = np.int0(boxi)
    points = boxi

    # We check whether the detected contour does not apply to the entire image and \
    # We perform a perspective transformation of the designated plate in order to straighten it
    for c in conts:
        x, y, w, h = cv.boundingRect(c)
        rectangled = cv.minAreaRect(c)
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.008 * peri, True)
        if len(conts) > 1:
            if 3 < len(approx) < 10:
                if w > 150 and h > 5 and w / h >= 1.5:
                    boxi = cv.boxPoints(rectangled)
                    area = w * h
                    boxi = np.int0(boxi)
                    if area_0 >= area > 200:
                        points = boxi
    cv.drawContours(resized, [points], 0, (0, 0, 255), 2)
    # The new vertex coordinates where the plate is pasted
    v1 = [0, 0]
    v2 = [600, 0]
    v3 = [0, 200]
    v4 = [600, 200]
    vertexes = np.float32([v1, v2, v4, v3])
    rect = np.zeros((4, 2), dtype="float32")
    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]
    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    # Making a final perspective transformation for license plate and saving it to the "dst" image
    M = cv.getPerspectiveTransform(rect, vertexes)
    dst = cv.warpPerspective(plates, M, (600, 220))

    return dst

def plate_processing(dst):
    # Conversion from BGR to HSV color space to remove the blue stripe with country mark
    hsv = cv.cvtColor(dst, cv.COLOR_BGR2HSV)
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([180, 255, 255])
    mask = cv.inRange(hsv, lower_blue, upper_blue)

    # Finding contours of the letters on the license plate and using hierarchy to remove inside outlines
    gray_dst = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
    numbers = cv.bilateralFilter(gray_dst, 11, 17, 17)
    numbers = cv.GaussianBlur(numbers, (5, 5), 0)
    adapt = cv.adaptiveThreshold(numbers, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 15, 2)
    adapt = cv.add(mask, adapt)
    cts, hierarchy = cv.findContours(adapt, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE, hierarchy=True)
    conts_list = []
    for i, h in enumerate(hierarchy[0]):
        if h[3] == -1:
            conts_list.append(cts[i])
    boxes = []
    boxes.clear()
    for c in conts_list:
        x, y, w, h = cv.boundingRect(c)
        ratio = h / w
        rect = cv.minAreaRect(c)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        if 40 < h < 200 and 10 < w < 200 and 1.2 < ratio < 5.5 and w * h > 3500:
            boxes.append(box)
    boxes = sorted(boxes, key=lambda boxes: boxes[0][0])
    # Returning list of boxes with detected letters, sorted by coordinates of their vertexes.
    return boxes


def letters_processing(boxes, dst):
    # Making perspective transformation of the letters to straighten them
    letters_list = []
    letters_list.clear()
    for box in boxes:
        v1 = [0, 0]
        v2 = [50, 0]
        v3 = [0, 100]
        v4 = [50, 100]
        vertexes = np.float32([v1, v2, v4, v3])
        rect = np.zeros((4, 2), dtype="float32")
        s = box.sum(axis=1)
        rect[0] = box[np.argmin(s)]
        rect[2] = box[np.argmax(s)]
        diff = np.diff(box, axis=1)
        rect[1] = box[np.argmin(diff)]
        rect[3] = box[np.argmax(diff)]
        M = cv.getPerspectiveTransform(rect, vertexes)
        letter = cv.warpPerspective(dst, M, (50, 100))

        # Appending transformed letters to the list
        letters_list.append(letter)

    # Checking how many contours were detected and rejecting those that are disturbances
    if len(boxes) < 7:
        for i in range(0, 7 - len(boxes)):
            letters_list.append(np.zeros((100, 50), dtype="float32"))
    if len(boxes) > 7:
        for i in range(0, len(boxes) - 7):
            del letters_list[-1]

    # Processing letters with morphological operations to remove noises and make negatives easier to detect
    adapt_letter = []
    adapt_letter.clear()
    for letter in letters_list:
        if letter.all() != 0:
            gray_letter = cv.cvtColor(letter, cv.COLOR_BGR2GRAY)
            number = cv.GaussianBlur(gray_letter, (5, 5), 0)
            _, number = cv.threshold(number, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
            kernel = np.ones((3, 3), np.uint8)
            number = cv.dilate(number, kernel, iterations=1)
            number = cv.erode(number, kernel, iterations=1)
            number = cv.dilate(number, kernel, iterations=2)
            number = cv.erode(number, kernel, iterations=2)
            number = 255 - number
            adapt_letter.append(number)
        else:
            adapt_letter.append(np.zeros((100, 50), dtype="uint8"))

    # A list containing the characters prepared for a final match
    return adapt_letter

def final_processing(adapt_letter, fonts_list):
    # Creating a list for final characters written to the JSON file
    final_list = []
    final_list.clear()

    # Array of reference letters related to a function called "fonts()"
    template = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','0','1','2','3','4','5','6','7','8','9','?']

    # Checking the best match for the cut letters and those from the reference list
    for original in adapt_letter:
        result = 0
        index = 0
        for i, font in enumerate(fonts_list):
            score = ssim(original, font, multichannel=False)
            if score > result:
                result = score
                index = i

        # If none of the letters were detected write '?' to the file
        if original.any() == 0:
            index = 36

        # Writing letter to the list containing final results
        final_list.append(template[index])

    return "".join(final_list)

