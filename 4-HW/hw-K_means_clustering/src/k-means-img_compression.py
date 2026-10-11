import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans

#-------------------------------- CAPTURE IMAGE FROM WEBCAM --------------------------------#
#-------------------------------------------------------------------------------------------#
def capture_from_webcam(delay_seconds=3):
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Cant open webcam. Check access!.")
    try:
        start_time = cv2.getTickCount()
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Cant read img from webcam.")

            frame = cv2.flip(frame, 1) # convert to mirror image
            elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            remaining = delay_seconds - int(elapsed)
            if remaining > 0:
                cv2.putText(
                    frame,
                    f"Shoot in {remaining} seconds...",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("Webcam - K-means image compression", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    raise RuntimeError("User cancelled the image capture.")
            else:
                if not cv2.imwrite(image_path, frame):
                    raise RuntimeError(f"Cant save image: {image_path}")
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        camera.release()
        cv2.destroyAllWindows()

image_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "webcam_capture.jpg",
)

img = capture_from_webcam()
print(img.shape) 
# number of possible colors of each pixel: 256^3 = 16,777,216
print("Img saved at:", image_path)
plt.imshow(img)
plt.axis("on")
plt.show()
#-------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------#

#-------------------------------- RESHAPE IMAGE  -------------------------------------------#
#-------------------------------------------------------------------------------------------#
X = img.reshape(
    img.shape[0] * img.shape[1],
    img.shape[2]
)

"""
1 sample = 1 pixel = 1 row of X = 1 data point = 1 RGB color
features = [R, G, B] = 3 columns of X
X = [data points, features] = [heght * width, 3]
"""
print("X.shape:", X.shape)
#-------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------#

K = 20

kmeans = KMeans(
    n_clusters=K, 
    random_state=0
    ).fit(X)
# ---> m1, m2, ..., mK = kmeans.cluster_centers_
print(kmeans.cluster_centers_)

print(kmeans.labels_)

labels = kmeans.predict(X) # label for any pixel (old or new --> ok)
# or: labels = kmeans.labels_ # label for data use after fit() (old data only)
print(labels.shape)
#print(labels[:100])

# empty array to store the compressed image
X_compressed = np.zeros_like(X)

for k in range(K):
    X_compressed[labels == k] = kmeans.cluster_centers_[k]

img_compressed = X_compressed.reshape(img.shape)

compressed_image_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "webcam_capture_compressed.jpg",
)
plt.imsave(compressed_image_path, img_compressed)
print("Img compressed saved at:", compressed_image_path)

plt.imshow(img_compressed)
plt.axis("on")
plt.show()

original_colors = np.unique(
    img.reshape(-1, 3),
    axis=0
)

compressed_colors = np.unique(
    img_compressed.reshape(-1, 3),
    axis=0
)

print("Number of colors in original image:", len(original_colors))
print("Number of colors in compressed image:", len(compressed_colors))