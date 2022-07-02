import cv2
import os

# A ML model built to identify faces - trained on large amounts of black/white images of faces
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Put all images of faces you wish to classify in this directory
folder_dir = "images"

# Goes through all images in the folder. Uses multiscale-detection, so the size
# of the image / size of the face within the image does not matter.
for image in os.listdir(folder_dir):
    img = cv2.imread("images/" + image)
    blackwhite_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_face_data.detectMultiScale(blackwhite_img)

    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    width = img.shape[1]
    height = img.shape[0]
    ratio = width/height

# Display the image with face detection (also normalizes the size)
    cv2.imshow('Face Detector', cv2.resize(img, (700, int(700/ratio))))

# Display the image until the window is closed - then display next image if there is one
    cv2.waitKey()

# Hopefully by this stage in the code all faces have been correctly identified :)
print('Code ran successfully')
