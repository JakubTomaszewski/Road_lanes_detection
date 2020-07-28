import cv2
import numpy as np
import keras


def convert_to_grayscale(image):
    """Converts an image to grayscale

    Parameters
    ----------
    image -- a numpy array represeting an image wth 3 color channels

    Returns
    -------
    image in grayscale without a color channel;
    """

    return cv2.cvtColor(image.astype('uint8') * 255, cv2.COLOR_BGR2GRAY)


def normalize_image(image):
    """Normalizes an image by dividing it by 255

    Parameters
    ----------
    image -- a numpy array representing an image

    Returns
    -------
    Normalized image
    """

    return image / 255


def histogram_equalization(image):
  """
  Parameters
  ----------

  image -- a numpy array representing an image in grayscale (without a color channel)

  Returns
  -------

  image of equalized grayscale
  """

  return cv2.equalizeHist(image)


def preprocess_image(image):
    """Applies grayscale transformation, equalization and normalization to an image

    Parameters
    ----------

    image -- a numpy array representing an image

    Returns
    -------
    preprocessed image

    """
    gray = convert_to_grayscale(image)
    equalized = histogram_equalization(gray)
    normalized = normalize_image(equalized)

    return normalized


def getCalssName(classNo):
    if   classNo == 0: return 'Speed Limit 20 km/h'
    elif classNo == 1: return 'Speed Limit 30 km/h'
    elif classNo == 2: return 'Speed Limit 50 km/h'
    elif classNo == 3: return 'Speed Limit 60 km/h'
    elif classNo == 4: return 'Speed Limit 70 km/h'
    elif classNo == 5: return 'Speed Limit 80 km/h'
    elif classNo == 6: return 'End of Speed Limit 80 km/h'
    elif classNo == 7: return 'Speed Limit 100 km/h'
    elif classNo == 8: return 'Speed Limit 120 km/h'
    elif classNo == 9: return 'No passing'
    elif classNo == 10: return 'No passing for vechiles over 3.5 metric tons'
    elif classNo == 11: return 'Right-of-way at the next intersection'
    elif classNo == 12: return 'Priority road'
    elif classNo == 13: return 'Yield'
    elif classNo == 14: return 'Stop'
    elif classNo == 15: return 'No vechiles'
    elif classNo == 16: return 'Vechiles over 3.5 metric tons prohibited'
    elif classNo == 17: return 'No entry'
    elif classNo == 18: return 'General caution'
    elif classNo == 19: return 'Dangerous curve to the left'
    elif classNo == 20: return 'Dangerous curve to the right'
    elif classNo == 21: return 'Double curve'
    elif classNo == 22: return 'Bumpy road'
    elif classNo == 23: return 'Slippery road'
    elif classNo == 24: return 'Road narrows on the right'
    elif classNo == 25: return 'Road work'
    elif classNo == 26: return 'Traffic signals'
    elif classNo == 27: return 'Pedestrians'
    elif classNo == 28: return 'Children crossing'
    elif classNo == 29: return 'Bicycles crossing'
    elif classNo == 30: return 'Beware of ice/snow'
    elif classNo == 31: return 'Wild animals crossing'
    elif classNo == 32: return 'End of all speed and passing limits'
    elif classNo == 33: return 'Turn right ahead'
    elif classNo == 34: return 'Turn left ahead'
    elif classNo == 35: return 'Ahead only'
    elif classNo == 36: return 'Go straight or right'
    elif classNo == 37: return 'Go straight or left'
    elif classNo == 38: return 'Keep right'
    elif classNo == 39: return 'Keep left'
    elif classNo == 40: return 'Roundabout mandatory'
    elif classNo == 41: return 'End of no passing'
    elif classNo == 42: return 'End of no passing by vechiles over 3.5 metric tons'


def main():
    # Creating cap instance
    cap = cv2.VideoCapture(0)
    four_cc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_file.mp4', four_cc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    # Initializing cap parameters
    frameWidth= 640         # CAMERA RESOLUTION
    frameHeight = 640
    brightness = 180
    threshold = 0.94         # PROBABLITY THRESHOLD
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Setting camera properties
    cap.set(3, frameWidth)
    cap.set(4, frameHeight)
    cap.set(10, brightness)

    # Loading model
    model = keras.models.load_model('best_model_second')
    # print(model.summary())

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            resized_img = cv2.resize(frame, (32, 32))
            preprocessed_img = preprocess_image(resized_img)
            # cv2.imshow('preprocessed', preprocessed_img)

            image = preprocessed_img.reshape(1, 32, 32, 1)
            cv2.putText(frame, "CLASS: ", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

            predictions = model.predict(image)
            class_idx = int(model.predict_classes(image))

            probab_value = np.max(predictions)
            if probab_value > threshold:
                cv2.putText(frame, str(class_idx)+" "+str(getCalssName(class_idx)), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, str(probab_value)+"%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
            # sleep(0.01)
            cv2.imshow('camera frame', frame)
            out.write(frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
