#!venv/bin/python

import cv2
import numpy as np
import os
import argparse
from keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenet
import tensorflow as tf
# Constants.
INPUT_WIDTH = 416
INPUT_HEIGHT = 416
SCORE_THRESHOLD = 0.5  # cls score
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.45  # obj confidence

# Text parameters.
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1

MODEL_JACKET = tf.keras.models.load_model("model_finetune0.88.h5")
MODEL_PANTS = tf.keras.models.load_model("mobilenetV2_full_trainable_pants_0.7638376355171204.h5")

CLASSES_JACKET = {0: "Invisible Jacket", 1: "With Jacket", 2: "No Jacket"}
CLASSES_PANTS = {0: "Invisible Pants", 1: "With Pants", 2: "No Pants"}

# Colors
BLACK = (0, 0, 0)
BLUE = (255, 178, 50)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
GEEN = (0, 255, 0)



def draw_label(input_image, label, left, top, color):
    """Draw text onto image at location."""

    # Get text size.
    dy = 25
    # Use text size to create a BLACK rectangle.
    for line, text in enumerate(label.split("\n")):
        text_size = cv2.getTextSize(text, FONT_FACE, FONT_SCALE, THICKNESS)
        dim, baseline = text_size[0], text_size[1]
        cv2.rectangle(input_image, (left,top - dim[1] - 2*line*dy ), (left + dim[0], top - 2*dim[1] - baseline), color, cv2.FILLED)
    # Display text inside the rectangle.
        cv2.putText(input_image, text, (left, top - dim[1] - line*dy), FONT_FACE, FONT_SCALE, BLACK, THICKNESS, cv2.LINE_AA)


def pre_process(input_image, net):
    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(input_image, 1 / 255, (INPUT_WIDTH, INPUT_HEIGHT), [0, 0, 0], 1, crop=False)

    # Sets the input to the network.
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers.
    output_layers = net.getUnconnectedOutLayersNames()
    outputs = net.forward(output_layers)
    # print(outputs[0].shape)

    return outputs


def get_color(jck, pnts):
    if jck == 2 and pnts == 2:
        return RED
    elif jck == 1 and pnts == 1:
        return GEEN
    else:
        return YELLOW


def getGuassianValue(W, H):
    x, y = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
    d = np.sqrt(x * x + y * y)
    sigma, mu = 1.0, 0.0
    g = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
    return g.T


def post_process(input_image, outputs):
    # Lists to hold respective values while unwrapping.
    class_ids = []
    confidences = []
    boxes = []
    heatmp = np.zeros(input_image.shape[:2], dtype=np.float64)
    # Rows.
    rows = outputs[0].shape[1]

    image_height, image_width = input_image.shape[:2]

    # Resizing factor.
    x_factor = image_width / INPUT_WIDTH
    y_factor = image_height / INPUT_HEIGHT

    # Iterate through 25200 detections.
    for r in range(rows):
        row = outputs[0][0][r]
        confidence = row[4]

        # Discard bad detections and continue.
        if confidence >= CONFIDENCE_THRESHOLD:
            classes_scores = row[5:]

            # Get the index of max class score.
            class_id = np.argmax(classes_scores)

            #  Continue if the class score is above threshold.
            if (classes_scores[class_id] > SCORE_THRESHOLD):
                confidences.append(confidence)
                class_ids.append(class_id)

                cx, cy, w, h = row[0], row[1], row[2], row[3]

                left = int((cx - w / 2) * x_factor)
                top = int((cy - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                box = np.array([left, top, width, height])
                boxes.append(box)

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        img = input_image[top:top + height, left:left + width]
        w, h = np.shape(heatmp[top:top + height, left:left + width])
        heatmp[top:top + height, left:left + width] += getGuassianValue(w, h)
        img = tf.image.resize(img, (100, 100))
        img = preprocess_mobilenet(img)
        pred_jacket = MODEL_JACKET.predict(img[None, ...])[0]
        pred_pants = MODEL_PANTS.predict(img[None, ...])[0]
        arg_jacket = np.argmax(pred_jacket)
        arg_pants = np.argmax(pred_pants)
        color = get_color(arg_jacket, arg_pants)
        cv2.rectangle(input_image, (left, top), (left + width, top + height), color, 3 * THICKNESS)
        label = "{}:{:.2f}\n{}:{:.2f}".format(CLASSES_JACKET[arg_jacket],
                                              pred_jacket[arg_jacket],
                                              CLASSES_PANTS[arg_pants],
                                              pred_pants[arg_pants])
        draw_label(input_image, label, left, top, color)

    return input_image, heatmp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default=None, help="Путь к вашему видео")
    parser.add_argument('--img', default=None, help="Путь к вашему изображению")
    parser.add_argument('--speed', default=1, help="Скорость обработки видео")
    args = parser.parse_args()

    video_path, img_path, speed = args.video, args.img, int(args.speed)

    # Load class names.
    model_path = "modelV2.onnx"
    # Give the weight files to the model and load the network using them.
    net = cv2.dnn.readNet(model_path)
    window_name = os.path.splitext(os.path.basename(model_path))[0]
    classes = ["person"]
    # newVideo = cv2.VideoWriter('detectVideo.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (1920,1080))
    # newVideoHeatMap = cv2.VideoWriter('HeatMap.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (1920,1080))
    # Load image.
    if video_path != None:
        rec = cv2.VideoCapture(video_path)

        heatmp = []
        iter = 0
        while rec.isOpened:
            ret, frame = rec.read()
            if iter % speed == 0:
                detections = pre_process(frame.copy(), net)
                img, htmp = post_process(frame.copy(), detections)
                heatmp.append(htmp)
                heatmapshow = None
                heatmapshow = cv2.normalize(np.sum(heatmp, axis=0), heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
                super_imposed_img = cv2.addWeighted(heatmapshow, 0.3, frame, 0.5, 0)
            # newVideo.write(img)
            # newVideoHeatMap.write(super_imposed_img)
            cv2.imshow("HeatMap", super_imposed_img)
            cv2.waitKey(1)
            cv2.imshow(window_name, img)
            cv2.waitKey(1)
        rec.release()
        # newVideo.release()
        # newVideoHeatMap.release()
        cv2.destroyAllWindows()

    elif img_path != None:
        frame = cv2.imread(img_path)
        detections = pre_process(frame.copy(), net)
        img, htmp = post_process(frame.copy(), detections)
        heatmapshow = None
        heatmapshow = cv2.normalize(htmp, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                    dtype=cv2.CV_8U)
        heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
        super_imposed_img = cv2.addWeighted(heatmapshow, 0.3, frame, 0.5, 0)
        cv2.imshow("HeatMap", super_imposed_img)
        cv2.imshow(window_name, img)
        cv2.waitKey(0)
