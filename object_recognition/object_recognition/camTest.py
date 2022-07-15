import cv2
import numpy as np


def capture_camera(cam_port: int) -> None:
    """Turn on webcam by camera port id
    and capture video from it. Could be stoped after using 'q' key."""

    cam = cv2.VideoCapture(cam_port)

    prototxt_path = "weights/deploy.prototxt.txt"
    model_path = "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"
    model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

    # Capture video from camera and split it into frames while cam is working
    while cam.isOpened():
        ret, frame = cam.read()

        h, w = frame.shape[:2]
        # preprocess the image: resize and performs mean subtraction
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (300, 300), (104.0, 177.0, 123.0)
        )

        # set the image into the input of the neural network
        model.setInput(blob)

        # perform inference and get the result
        output = np.squeeze(model.forward())
        font_scale = 1.0
        for i in range(0, output.shape[0]):
            # get the confidence
            confidence = output[i, 2]

            # if confidence is above 50%, then draw the surrounding box
            if confidence > 0.5:

                # get the surrounding box cordinates and upscale them
                # to original image
                box = output[i, 3:7] * np.array([w, h, w, h])

                # convert to integers
                start_x, start_y, end_x, end_y = box.astype(np.int)

                # draw the rectangle surrounding the face
                cv2.rectangle(
                    frame,
                    (start_x, start_y),
                    (end_x, end_y),
                    color=(255, 0, 0),
                    thickness=2,
                )

                # draw text as well
                cv2.putText(
                    frame,
                    f"{confidence*100:.2f}%",
                    (start_x, start_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 0, 0),
                    2,
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
