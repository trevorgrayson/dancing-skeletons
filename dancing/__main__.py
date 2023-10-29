from . import *


def skeletons(img, lmList):
    skull = Image.open("skull.png").convert('RGBA')

    shoulder_l = lmList[12][1:]
    eye_l = lmList[5][1:]
    mouth = lmList[10][1:]
    ear_l = lmList[8][1:]
    ear_r = lmList[7][1:]
    head_height = floor((ear_r[0] - ear_l[0]) * 1.2)
    head_width = floor((shoulder_l[1] - eye_l[1]))
    if head_width < 1 or head_height < 1:  # did you turn around?
        return  # bail out

    skull_resized = skull.resize((head_height,head_width))
    img.paste(skull_resized, (ear_l[0] - floor(head_width*0.1),
                              ear_l[1] - floor(head_height*0.7)), mask=skull_resized)  # 100 is half of skull height


def human_detected(lmList):
    return len(lmList) > 8


def main():
    detector = poseDetector()
    cap = cv2.VideoCapture(0)

    cv2.namedWindow("dance", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("dance", 2080, 1420)

    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.getPosition(img)
        img = Image.fromarray(img)
        # detector.showFps(img)

        if human_detected(lmList):
            pass
            skeletons(img, lmList)

        cv2.imshow("dance", np.array(img))

        cv2.waitKey(1)


if __name__ == "__main__":
    main()
