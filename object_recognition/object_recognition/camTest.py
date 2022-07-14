import cv2


def capture_camera(cam_port: int) -> None:
    """Turn on webcam by camera port id
    and capture video from it. Could be stoped after using 'q' key."""

    cam = cv2.VideoCapture(cam_port)

    while cam.isOpened():
        ret, frame = cam.read()
        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()


def main() -> None:
    capture_camera(3)
    cv2.destroyAllWindows()


if __name__ == "__main__":

    main()
