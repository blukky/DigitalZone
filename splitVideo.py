
import cv2
import os



movies = ["movies2.mp4", "movies3.mp4"]


def splitMovie(path_to_movie):
    folder = ".".join(path_to_movie.split(".")[:-1])
    if not os.path.exists(folder):
        os.mkdir(folder)


    rec = cv2.VideoCapture(path_to_movie)

    i = 0
    while rec.isOpened() and i < rec.get(cv2.CAP_PROP_FRAME_COUNT):
        ret, frame = rec.read()
        if i % 5 == 0 and ret:
            print("Iter: " + str(i) + " for " + str(rec.get(cv2.CAP_PROP_FRAME_COUNT)))
            cv2.imwrite(f"{folder}/{folder}_image_{i}.jpg", frame)
        i += 1
    rec.release()



for movie in movies:
    splitMovie(movie)