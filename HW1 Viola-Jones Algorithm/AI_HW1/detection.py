import os
import cv2
import matplotlib.pyplot as plt

def detect(dataPath, clf):
    """
    Please read detectData.txt to understand the format. Load the image and get
    the face images. Transfer the face images to 19 x 19 and grayscale images.
    Use clf.classify() function to detect faces. Show face detection results.
    If the result is True, draw the green box on the image. Otherwise, draw
    the red box on the image.
      Parameters:
        dataPath: the path of detectData.txt
      Returns:
        No returns.
    """
    # Begin your code (Part 4)
    """
    Read all data in the txt file and find the image name to read the image.
    Read all boxes information , capture them in the original image, resize 
    them to 19 x 19 , and classify them respectively. If the return value of 
    classify() is 1, it is a face, and make a green box on the original image.
    Otherwise, make a red one.

    Variables:
      f:
        Used to open the file and read all data in the file
      file :
        Used to store the data in the file
      index:
        Used to find the last '/' and find the root path
      picPath:
        The path to the certain image
      pic_classify:
        Used to be classified, so it need to be read in gray scale
      pic:
        The original picture, it is read in color
      n:
        The nums of the boxes in the picture
      face:
        The box information that include the face should be classified
      img:
        The image that include the face captured from the pic_classify
        
    """
    f = open(dataPath, 'r')
    file = f.read().split()
    f.close()
    index = 0
    for i in range(len(dataPath)):
      if dataPath[i] == '/':
        index = i + 1
    dataPath = dataPath[:index]
    for i in range(len(file)):
      if (file[i][-4:] == '.jpg'): 
        picPath = dataPath + file[i]
        pic_classify = cv2.imread(picPath, cv2.IMREAD_GRAYSCALE)
        pic  = cv2.imread(picPath, cv2.IMREAD_COLOR)
        i += 1
        n = int(file[i])
        i += 1
        for _ in range(n):
          face = file[i:i+4]
          for num in range(len(face)):
            face[num] = int(face[num])
          i += 4
          img = pic_classify[face[1]:face[1] + face[2], face[0]:face[0] + face[3]]
          if img is not None:
            img = cv2.resize(img, (19,19))
          if clf.classify(img):
            cv2.rectangle(pic, (face[0], face[1]), (face[0] + face[3], face[1] + face[2]), (0, 255, 0), 2)
          else:
            cv2.rectangle(pic, (face[0], face[1]), (face[0] + face[3], face[1] + face[2]), (0, 0, 255), 2)
          img = None
        cv2.imshow('img', pic)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # End your code (Part 4)
