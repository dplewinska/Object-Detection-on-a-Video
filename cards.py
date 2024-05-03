import cv2 as cv2
import numpy as np
from os import listdir
from os.path import isfile, join

def get_card_cords(frame):
    templates = [f for f in listdir("./data/CVP2-data/cards/templatesjpg") if isfile(join("./data/CVP2-data/cards/templatesjpg", f))]
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    matches = []
    for t in templates:
        path = "./data/CVP2-data/cards/templatesjpg/" + str(t)
        # if path != "./data/CVP2-data/cards/templatesjpg/IMG_2119.jpg":
        #     continue
        template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        w, h = template.shape[::-1]
        template = cv2.resize(template, (int(w/18), int(h/18)))
        w, h = template.shape[::-1]
        res = cv2.matchTemplate(frame_gray, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val > 0.6:
            top_left = max_loc
            if top_left[0] < 700 or top_left[0] > 1000 or top_left[1] < 1000 or top_left[0] > 1300:
                continue
            bottom_right = (top_left[0] + w, top_left[1] + h)
            # top_left = (top_left[0] - 100, top_left[1] - 50)
            # bottom_right = (bottom_right[0] + 50, bottom_right[1] + 50)
            matches.append((max_val, top_left, bottom_right))
    matches = sorted(matches, key=lambda x: x[0], reverse=True)
    print(matches)
    # print(matches)
    if len(matches) > 0:
        return matches[0][1], matches[0][2]
    
    return None, None