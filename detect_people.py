#!venv/bin/python

import cv2
import numpy as np
import os
import argparse

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

# Colors
BLACK = (0, 0, 0)
BLUE = (255, 178, 50)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)


# PATH_SAVE = "../classifier_data/"

def draw_label(input_image, label, left, top):
    """Draw text onto image at location."""

    # Get text size.
    text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
    dim, baseline = text_size[0], text_size[1]
    # Use text size to create a BLACK rectangle.
    cv2.rectangle(input_image, (left, top), (left + dim[0], top + dim[1] + baseline), BLACK, cv2.FILLED)
    # Display text inside the rectangle.
    cv2.putText(input_image, label, (left, top + dim[1]), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS, cv2.LINE_AA)


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


def post_process(input_image, outputs):
    # Lists to hold respective values while unwrapping.
    class_ids = []
    confidences = []
    boxes = []
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
        cv2.rectangle(input_image, (left, top), (left + width, top + height), BLUE, 3 * THICKNESS)
        label = "{}:{:.2f}".format(classes[class_ids[i]], confidences[i])
        draw_label(input_image, label, left, top)

    return input_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default=None, help="Путь к вашему видео")
    parser.add_argument('--img', default=None, help="Путь к вашему изображению")
    args = parser.parse_args()

    video_path, img_path = args.video, args.img
    # Load class names.
    model_path = "modelV2.onnx"
    # Give the weight files to the model and load the network using them.
    net = cv2.dnn.readNet(model_path)
    window_name = os.path.splitext(os.path.basename(model_path))[0]
    classes = ["person"]

    # Load image.
    if video_path != None:
        rec = cv2.VideoCapture(video_path)


        while rec.isOpened:
            ret, frame = rec.read()
            detections = pre_process(frame.copy(), net)
            img = post_process(frame.copy(), detections)
            cv2.imshow(window_name, img)
            cv2.waitKey(1)
        rec.release()
        cv2.destroyAllWindows()
    elif img_path != None:
        frame = cv2.imread(img_path)
        detections = pre_process(frame.copy(), net)
        img = post_process(frame.copy(), detections)

        cv2.imshow(window_name, img)
        cv2.waitKey(0)