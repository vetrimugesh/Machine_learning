from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np

import imutils
import cv2
import matplotlib.pyplot as plt

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

width = 2.5

img = cv2.imread('size.jpeg')


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray,cmap='gray')
plt.show()

gray = cv2.GaussianBlur(gray, (1, 1), 0)

edged = cv2.Canny(gray, 50, 100)

plt.imshow(edged,cmap='gray')
plt.show()
edged = cv2.dilate(edged, None, iterations=1)
plt.imshow(edged,cmap='gray')
plt.show()
edged = cv2.erode(edged, None, iterations=1)
plt.imshow(edged,cmap='gray')
plt.show()

cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
cv2.CHAIN_APPROX_SIMPLE)

cnts = imutils.grab_contours(cnts)
print(cnts)

(cnts, _) = contours.sort_contours(cnts)
pixelsPerMetric = None

orig = img

for c in cnts:
    if cv2.contourArea(c) < 100:
        continue
    box = cv2.minAreaRect(c)
    box = cv2.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")

    box = perspective.order_points(box)
    cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 1)
    for (x, y) in box:
        cv2.circle(orig, (int(x), int(y)), 2, (0, 0, 255), 1)
    
    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    cv2.circle(orig, (int(tltrX), int(tltrY)), 1, (255, 0, 0), -1)
    cv2.circle(orig, (int(blbrX), int(blbrY)), 1, (255, 0, 0), -1)
    cv2.circle(orig, (int(tlblX), int(tlblY)), 1, (255, 0, 0), -1)
    cv2.circle(orig, (int(trbrX), int(trbrY)), 1, (255, 0, 0), -1)
    
    cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
        (255, 0, 255), 1)
    cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
        (255, 0, 255), 1)

    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
    
    if pixelsPerMetric is None:
        pixelsPerMetric = dB / width
    

    dimA = dA / pixelsPerMetric
    dimB = dB / pixelsPerMetric

    cv2.putText(orig, "{:.1f}cm".format(dimA),
        (int(tltrX - 15), int(tltrY -10 )), cv2.FONT_HERSHEY_SIMPLEX,
        0.2, (255, 0, 0), 1)
    cv2.putText(orig, "{:.1f}cm".format(dimB),
        (int(trbrX - 5), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
        0.2, (255, 0, 0), 1)
plt.imshow(orig,cmap='gray')
plt.show()
