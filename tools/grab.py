import cv2, sys
for dev in sys.argv[1:]:
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    ok = False
    if cap.isOpened():
        for _ in range(15): ok, f = cap.read()
    if ok and f is not None and f.ndim == 3 and f.std() > 3:
        out = "frame_" + dev.split("/")[-1] + ".jpg"
        cv2.imwrite(out, cv2.resize(f, (480, int(480 * f.shape[0] / f.shape[1]))))
        print(dev, "OK", f.shape)
    else:
        print(dev, "no usable frame")
    cap.release()
