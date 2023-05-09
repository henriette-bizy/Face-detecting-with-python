import cv2
image = cv2.imread('./pictures/picture1.jpg')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
detected_faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
for (x, y, w, h) in detected_faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cropped_img = image[y:y+h, x:x+w]
    cv2.imwrite('./pictures/croppedPicture.jpg', cropped_img)

cv2.imshow('Faces Detected', image)
cv2.waitKey(0)
cv2.destroyAllWindows()