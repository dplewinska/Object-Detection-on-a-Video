
import numpy as np
import cv2
from utils import *
from board import *
from cards import *
from PIL import Image
# tracker = cv2.TrackerMedianFlow_create()
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ------------------------------------------------------------------------------------------------------
'''
        PROJECT 2 COMPUTER VISION - tracking Kolejka board game
        authors: Weronika Zawadzka 151943 & Dominika Plewińska 151929
        date: 19.12.2023
'''
# ------------------------------------------------------------------------------------------------------

# FPS: 30frames/second
# check for board update every 3 sec = 90 frames?


def prepare_data():
    # prepareing the data
    settings = load_settings("./data/settings.json")
    videos = read_videos(download=False)
    if not videos:
        print("No videos found.")
    reference_board, reference_bboxes, reference_bboxes_names = get_reference_board()
    # print(f"Reference board size: {reference_board.shape}")
    # areas = {name: reference_board[y:y+h, x:x+w]
    #          for (name, (x, y, w, h)) in zip(reference_bboxes_names, reference_bboxes)}
    return settings, videos, reference_board, reference_bboxes, reference_bboxes_names


def get_day_of_the_week(image, x):
    '''
        image - roi bounding box image
        x - of the averaged bounding boxes

    '''
    days = {"0": "Monday", "1": "Tuesday",
            "2": "Wednesday", "3": "Thursday", "4": "Friday"}
    w = image.shape[1]
    bins = list(range(0, w, w//5))  # like [0, 49, 98, 147, 196, 245]

    for i in range(len(bins)-1, 0, -1):
        if x <= bins[i] and x >= bins[i-1]:
            return days[str(i-1)]

    return None


def test_bazar(roi):
    lab_image = cv2.cvtColor(roi, cv2.COLOR_BGR2Lab)
    # calulate mean
    mean_color = np.mean(lab_image, axis=(0, 1))

    lower_color = np.array([mean_color[0]-40, 128, 128])
    upper_color = np.array([mean_color[0]+20, 255, 255])
    # print(f"Lower_color = {lower_color}, upper: {upper_color}")

    # lower_color = np.array([100, 128, 128])
    # upper_color = np.array([120, 255, 255])

    # Create a binary mask using the adjusted color range
    pawn_mask = cv2.inRange(lab_image, lower_color, upper_color)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(
        pawn_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around the detected pawns
    xs = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # print(f"coords: {x}, {y}")
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
        xs.append(x)

    x = np.mean(xs)
    day = get_day_of_the_week(roi, x)
    # print(f"DAY OF THE WEEK: {day}")
    # cv2.putText(roi, day, (150, 30),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 5)
    # cv2.imshow('BAZAR ROI::', roi)
    return day


def visualize_frame(frame, title, rec=None):
    f = frame.copy()
    if type(rec) != type(None):
        cv2.drawContours(f, [rec], -1, (0, 0, 255), 2)
    cv2.imshow(title, cv2.resize(
        f, (0, 0), fx=0.3, fy=0.3))


def find_reference_bbox(template, img, threshold=0.8):
    ''' https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html '''
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()  # as scale invariant

    # Find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(template, None)
    kp2, des2 = sift.detectAndCompute(img, None)

    # Create BFMatcher (Brute-Force Matcher)
    bf = cv2.BFMatcher()

    # Match descriptors using KNN
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good_matches = [m[0]
                    for m in matches if m[0].distance < threshold * m[1].distance]

    # # Draw matches on the images
    # img_matches = cv2.drawMatches(template, kp1, img, kp2, good_matches,
    #                               None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Extract matched keypoints
    src_pts = np.float32(
        [kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Find the homography matrix
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    h, w = template.shape
    template_corners = np.float32(
        [[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    target_corners = cv2.perspectiveTransform(template_corners, M)
    x, y, w, h = cv2.boundingRect(target_corners)
    # img_with_bbox = img.copy()
    # cv2.rectangle(img_with_bbox, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # cv2.imshow('bbbb', cv2.resize(
    #     img_with_bbox, (0, 0), fx=0.2, fy=0.2))
    if w < 100 or h < 100:
        return None
    return (y, x, w, h)


def main():
    # prepareing the data
    settings, videos, reference_board, reference_bboxes, reference_bboxes_names = prepare_data()
    bbox_rois = {name: reference_board[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] for (name, bbox) in zip(
        *[reference_bboxes_names, reference_bboxes])}
    offset = 30
    # looping through the videos frames
    for video in videos:
        board_history = []
        day_history = []
        board_coords = None  # initialize board coordinates
        frame_c = 0  # initialize frame count
        bazar_bbox = None
        while True:
            # ----------------- read frame -------------------------
            ret, frame = video.read()
            frame = cv2.imread("testframe3.png", cv2.IMREAD_COLOR)
            if not ret:
                break
            frame_c += 1
            visualize_frame(frame, 'Original frame', board_coords)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            # update every 10 frames = 1/3 sec (FPS: 30frames/second)
            if (frame_c-1) % 30 != 0:
                continue
            # ------------------ board updates -----------------------
            # we have 5 board recalulations at the start and choose the median of them
            if type(board_coords) == type(None) or (frame_c-1) in list(range(0, 61, 10)):
                # try to calulate board coordinates by finding largest rectangle
                bc = get_board_coords(frame)
                board_history.append(bc)
                filtered_board_history = [
                    item for item in board_history[-10:] if item is not None]
                if len(filtered_board_history) == 0:
                    continue
                stacked_arrays = np.stack(filtered_board_history, axis=-1)
                new_board_coords = np.median(
                    stacked_arrays, axis=-1).astype(int)
                if type(new_board_coords) != type(None):
                    board_coords = new_board_coords
                continue

            # # --------------- transform board perspective -----------
            frame = perspective_transform(
                frame, board_coords, reference_board)
            # # ---------------------- bazar --------------------------
            if bazar_bbox is None:
                bazar_bbox = find_reference_bbox(bbox_rois['bazar'], frame)
            else:
                y, x, w, h = bazar_bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                day = test_bazar(frame[y:y+h, x:x+w])
                day_history.append(day)

                day_history = day_history[-50:]
                cv2.putText(frame[y:y+h, x:x+w], max(set(day_history), key=day_history.count), (150, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 5)
            # ---------------------- queues -------------------------
            # for (bbox, class_name) in zip(*[reference_bboxes, reference_bboxes_names]):
            #     x, y, w, h = bbox
            #     x, y, w, h = map(
            #         int, (x, max(y - offset, 0), min(frame.shape[1] - x, w + offset), min(frame.shape[0] - y, h + offset)))
            #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # -------------------------------------------------------
                
            # # ---------------------- cards -------------------------
            top_left, bottom_right = get_card_cords(frame)
            print(top_left)
            if top_left is not None:
                cv2.rectangle(frame, top_left, bottom_right, 255, 2)

            cv2.imshow('Board View', cv2.resize(frame, (0, 0), fx=0.3, fy=0.3))

    for video in videos:
        video.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
