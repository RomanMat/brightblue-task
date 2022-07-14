import cv2


def capture_camera(cam_port: int) -> None:
    """Turn on webcam by camera port id
    and capture video from it. Could be stoped after using 'q' key."""

    cam = cv2.VideoCapture(cam_port)

    # initialize the face recognizer (default face haar cascade)
    face_cascade = cv2.CascadeClassifier(
        "cascades/haarcascade_frontalface_default.xml"
    )

    # Capture video from camera and split it into frames while cam is working
    while cam.isOpened():
        ret, frame = cam.read()

        # Convering each frame to gray color
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, 2, 5)

        # for every face, draw a blue rectangle
        for x, y, width, height in faces:
            cv2.rectangle(
                frame,
                (x, y),
                (x + width, y + height),
                color=(255, 0, 0),
                thickness=2,
            )

        cv2.imshow("Camera", frame)

        # Waiting for "q" key to be pressed, if pressed - stop transletion
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()


def main() -> None:
    capture_camera(2)
    cv2.destroyAllWindows()


if __name__ == "__main__":

    main()
