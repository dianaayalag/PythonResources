# Rotate image correctly
import imutils

def rotateImage(img, angle):
  img_rotated = imutils.rotate_bound(img, angle)
  return img_rotated
  
#########################################################################################################################
  
# Resize image from width or heigh, with ratio:
import cv2

def resize(img, width=None, height=None, interpolation=cv2.INTER_AREA):
    h, w, c = img.shape
    global ratio

    if width is None and height is None:
        return img
    elif width is None:
        ratio = height / h
        width = int(w * ratio)
        resized = cv2.resize(img, (width, height), interpolation)
        return resized
    else:
        ratio = width / w
        height = int(h * ratio)
        resized = cv2.resize(img, (width, height), interpolation)
        return resized
        
#########################################################################################################################

# Get euclidian distance from two points:
import numpy as np

def dist(x,y):
    return np.sqrt(np.sum((x-y)**2))
    
# Find rectangle corners in a photo (localize biggest rectangle):
# import cv2

#########################################################################################################################

def getPoints(image_resized):
    gray_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    image_blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edged = cv2.Canny(image_blurred, 75, 200)
    corners = cv2.goodFeaturesToTrack(edged, 100, 0.01, 10)
    img, contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]
    for con in contours:
        peri = cv2.arcLength(con, True)
        approx = cv2.approxPolyDP(con, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break
    return screenCnt
   
#########################################################################################################################
    
# Perspective warp from screenCnt:
# import cv2
# import numpy as np

def getPts1(screenCnt):
    if (dist(screenCnt[0][0], screenCnt[1][0]) > dist(screenCnt[1][0], screenCnt[2][0])):
        pts1 = np.float32([screenCnt[0][0], screenCnt[1][0], screenCnt[3][0], screenCnt[2][0]])
    else:
        pts1 = np.float32([screenCnt[1][0], screenCnt[2][0], screenCnt[0][0], screenCnt[3][0]])
    return pts1
    
 def getParams(pts1, image_resized):
    w, h, c = image_resized.shape
    if w < h:
        pts2 = np.float32([[0,0],[0,h],[w,0],[w,h]])
        test = cv2.getPerspectiveTransform(pts1,pts2)
        warp = cv2.warpPerspective(image_resized,test,(w,h))
    else:
        pts2 = np.float32([[0,0],[0,w],[h,0],[h,w]])
        test = cv2.getPerspectiveTransform(pts1,pts2)
        warp = cv2.warpPerspective(image_resized,test,(h,w))
    return warp

#########################################################################################################################

# Threshold image:
# import cv2

def threshold_image(img, th):
  img_grey_th = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
  test, th_image = cv2.threshold(img_grey_th, th, 255, cv2.THRESH_BINARY)
  return th_image

#########################################################################################################################

# Template Matching
# import cv2
# import numpy as np

def match_templates(img, route)
  templates = [cv2.imread(file, 0) for file in glob.glob("static/" + route + "/*.jpg")]
  for template in templates:
      w, h = template.shape[::-1]
      res = cv2.matchTemplate(th2, template, cv2.TM_CCOEFF_NORMED)
      threshold = 0.64
      loc = np.where(res >= threshold)

      for pt in zip(*loc[::-1]):
          cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)
  return img

#########################################################################################################################

# Draw Rectangle on Image:
# import cv2

def rectangle(img, pts):
  cv2.rectangle(img, pt[0], pt[1], (0, 255, 255), 2)
  return img
  


