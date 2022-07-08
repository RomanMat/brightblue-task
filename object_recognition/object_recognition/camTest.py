import cv2 as cv

# Read img and show it for endless amount of time

# img = cv.imread("imgs/man_1.jpg")
# cv.imshow("Man #1",img)
# cv.waitKey(0)


captcure = cv.VideoCapture(0) 
while True:
    isTrue, frame = captcure.read()
    cv.imshow("Camera #0", frame)
    if cv.waitKey(20) & 0xFF==ord("c"):
        break

captcure.release()
cv.destroyAllWindows()