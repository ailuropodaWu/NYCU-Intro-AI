import os
import cv2


def loadImages(dataPath):
    """
    load all Images in the folder and transfer a list of tuples. The first 
    element is the numpy array of shape (m, n) representing the image. 
    The second element is its classification (1 or 0)
      Parameters:
        dataPath: The folder path.
      Returns:
        dataset: The list of tuples.
    """
    # Begin your code (Part 1)
    """
    Read all folder under the dataPath if the folder name is 'face' 
    labeled all images under the folder by 1, if the name is 'non-face',
    labeled images by 0. And read all images under the folder. Then make 
    the tuple (image, label) and append to dataset.

    Variables:
    folders:
      All folder under the dataPath
    label:
      To label the image face or non-face
    imgPaths:
      The paths to all images under the folder individually
    img:
      The image read from the imgPath
    data:
      The tuple of the image and the label
    dataset:
      All datas
    """     
    folders = os.listdir(dataPath)
    dataset = list()
    for folder in folders:
      if folder == 'face':
        label = 1
      elif folder == 'non-face':
        label = 0
      folderPath = dataPath + '/' + folder
      imgPaths = os.listdir(folderPath)
      for imgPath in imgPaths :
        imgPath = folderPath + '/' + imgPath
        img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
        data = (img, label)
        dataset.append(data)
    # End your code (Part 1)
    return dataset
