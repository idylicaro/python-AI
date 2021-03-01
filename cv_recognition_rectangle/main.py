import cv2

camera = cv2.VideoCapture(0)


def redim(img, width):
    height = int(img.shape[0] / img.shape[1] * width)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return img


while True:
    (success, frame) = camera.read()

    if not success:
        break

    # frame = redim(frame, 320)
    frame_pb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('Grey frame', frame_pb)

    _, bin = cv2.threshold(frame_pb, 90, 255, cv2.THRESH_BINARY)
    # cv2.imshow('binary frame', bin)

    blur = cv2.GaussianBlur(bin, (5, 5), 0)
    cv2.imshow('Bluerd frame', blur)

    (contours, hierarchy) = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # clear not closed contours
    for c in contours:
        perimeter = cv2.arcLength(c, True)
        if perimeter > 40:
            approach = cv2.approxPolyDP(c, 0.03 * perimeter, True)
            if 2 < len(approach) < 8:
                (x, y, width, height) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + height, y + width), (255, 0, 0), 2)

    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
    #cv2.imshow('Main Frame', redim(frame, 640))
    cv2.imshow('Main Frame', frame)
    if cv2.waitKey(1) and 0xFF == ord('a'):
        break

cv2.release()
cv2.destroyAllWindows()
