import cv2


def redim(img, width):
    height = int(img.shape[0] / img.shape[1] * width)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return img


df = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

camera = cv2.VideoCapture(0)

while True:
    (success, frame) = camera.read()

    if not success:
        break

    frame = redim(frame, 320)
    frame_pb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = df.detectMultiScale(frame_pb, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
    frame_temp = frame.copy()
    for (x, y, width, height) in faces:
        cv2.rectangle(frame_temp, (x, y), (x + width, y + height), (255, 0, 0), 2)
    cv2.imshow('Recognition', redim(frame_temp, 640))
    
    if cv2.waitKey(1) and 0xFF == ord('S'):
        break

camera.release()
cv2.destroyAllWindows()
