import cv2
import numpy as np
import utlis
import subprocess

try:
    import cv2
except ImportError:
    print("OpenCV is not installed. Installing...")
    try:
        subprocess.run(["pip", "install", "opencv-python"])
        import cv2  # Attempt to import again after installation
        print("OpenCV installed successfully. Version:", cv2.__version__)
    except Exception as e:
        print("Error installing OpenCV:", str(e))
except Exception as e:
    print("An error occurred:", str(e))

def initialize_trackbars():
    cv2.namedWindow("Trackbars")
    cv2.createTrackbar("Threshold1", "Trackbars", 50, 255, lambda x: None)
    cv2.createTrackbar("Threshold2", "Trackbars", 150, 255, lambda x: None)

def val_trackbars():
    threshold1 = cv2.getTrackbarPos("Threshold1", "Trackbars")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Trackbars")
    return threshold1, threshold2

def biggest_contour(contours):
    max_area = 0
    biggest_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            biggest_contour = contour
    return biggest_contour, max_area

def reorder(points):
    if points is not None and points.size == 8:
        points = points.reshape((4, 2))
        new_points = np.zeros((4, 1, 2), dtype=np.int32)
        add = points.sum(1)
        new_points[0] = points[np.argmin(add)]
        new_points[3] = points[np.argmax(add)]
        diff = np.diff(points, axis=1)
        new_points[1] = points[np.argmin(diff)]
        new_points[2] = points[np.argmax(diff)]
        return new_points
    else:
        return None

def draw_rectangle(img, points, thickness=2):
    if points is not None:
        pts = points.reshape((4, 2))
        cv2.polylines(img, [pts.astype(int)], isClosed=True, color=(0, 255, 0), thickness=thickness)
    return img

def stack_images(img_array, scale, labels):
    rows = len(img_array)
    cols = len(img_array[0])
    rows_available = isinstance(img_array[0], list)
    width = img_array[0][0].shape[1]
    height = img_array[0][0].shape[0]

    if rows_available:
        for x in range(0, rows):
            for y in range(0, cols):
                if img_array[x][y].shape[:2] == img_array[0][0].shape[:2]:
                    img_array[x][y] = cv2.resize(img_array[x][y], (0, 0), None, scale, scale)
                else:
                    img_array[x][y] = cv2.resize(img_array[x][y], (img_array[0][0].shape[1], img_array[0][0].shape[0]), None, scale, scale)
                if len(img_array[x][y].shape) == 2:
                    img_array[x][y] = cv2.cvtColor(img_array[x][y], cv2.COLOR_GRAY2BGR)
        image_blank = np.zeros((height, width, 3), np.uint8)
        hor = [image_blank] * rows
        hor_con = [image_blank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(img_array[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if img_array[x].shape[:2] == img_array[0].shape[:2]:
                img_array[x] = cv2.resize(img_array[x], (0, 0), None, scale, scale)
            else:
                img_array[x] = cv2.resize(img_array[x], (img_array[0].shape[1], img_array[0].shape[0]), None, scale, scale)
            if len(img_array[x].shape) == 2:
                img_array[x] = cv2.cvtColor(img_array[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(img_array)
        ver = hor
    return ver

# Choose between camera and image mode
use_camera_str = input("Enter True for camera and False for pre-saved Image").title()
use_camera = use_camera_str == "True"


if use_camera:
    # Set up video capture
    cap = cv2.VideoCapture(0)  # set 1 for an external camera and 0 for an internal camera
    cap.set(10, 160)
else:
    img = cv2.imread("1.jpg")
    if img is None:
        print("Error: Unable to load the image.")
        exit()

# Initialize trackbars
initialize_trackbars()
heightImg = 640
widthImg = 480
count = 0
imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)  # CREATE A BLANK IMAGE FOR TESTING DEBUGING IF REQUIRED

while True:
    if use_camera:
        success, img = cap.read()
        if not success:
            print("Error: Unable to capture video.")
            break  # Exit the loop or handle the error accordingly

    img = cv2.resize(img, (widthImg, heightImg))  # Resize image

    # Convert image to grayscale
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)

    # Get Canny edges
    thresholds = val_trackbars()
    imgThreshold = cv2.Canny(imgBlur, thresholds[0], thresholds[1])

    # Apply dilation and erosion
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)

    # Find contours
    imgContours = img.copy()
    imgBigContour = img.copy()
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Find the biggest contour
    # Find the biggest contour
    biggest, maxArea = utlis.biggestContour(contours)  # FIND THE BIGGEST CONTOUR
    if biggest is not None and biggest.size != 0:
        biggest = utlis.reorder(biggest)  # reorder only if it's not None

        # Ensure 'biggest' is not None before attempting to draw contours
        if biggest is not None:
            cv2.drawContours(imgBigContour, [biggest], -1, (0, 255, 0), 20)  # DRAW THE BIGGEST CONTOUR
            imgBigContour = utlis.drawRectangle(imgBigContour, biggest, 2)
            pts1 = np.float32(biggest)  # PREPARE POINTS FOR WARP
            pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])  # PREPARE POINTS FOR WARP
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

            # Crop and resize warped image
            imgWarpColored = imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]
            imgWarpColored = cv2.resize(imgWarpColored, (widthImg, heightImg))

            # Apply adaptive threshold
            imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
            imgAdaptiveThre = cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
            imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
            imgAdaptiveThre = cv2.medianBlur(imgAdaptiveThre, 3)

            # Display results
            imageArray = [[img, imgGray, imgThreshold, imgContours],
                          [imgBigContour, imgWarpColored, imgWarpGray, imgAdaptiveThre]]
            lables = [["Original", "Gray", "Threshold", "Contours"],
                      ["Biggest Contour", "Warp Prespective", "Warp Gray", "Adaptive Threshold"]]

            stackedImage = stack_images(imageArray, 0.75, lables)
            cv2.imshow("Result", stackedImage)

            # Save image when 's' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('s'):
                cv2.imwrite("myImage" + str(count) + ".jpg", imgWarpColored)
                cv2.rectangle(stackedImage,
                              ((int(stackedImage.shape[1] / 2) - 230), int(stackedImage.shape[0] / 2) + 50),
                              (1100, 350), (0, 255, 0), cv2.FILLED)
                cv2.putText(stackedImage, "Scan Saved",
                            (int(stackedImage.shape[1] / 2) - 200, int(stackedImage.shape[0] / 2)),
                            cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
                cv2.imshow('Result', stackedImage)
                cv2.waitKey(300)
                count += 1
    else:
        # If no contour is found, display blank image
        imageArray = [[img, imgGray, imgThreshold, imgContours],
                      [imgBlank, imgBlank, imgBlank, imgBlank]]
        lables = [["Original", "Gray", "Threshold", "Contours"],
                  ["Biggest Contour", "Warp Prespective", "Warp Gray", "Adaptive Threshold"]]

        stackedImage = stack_images(imageArray, 0.75, lables)
        cv2.imshow("Result", stackedImage)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
if use_camera:
    cap.release()
cv2.destroyAllWindows()
