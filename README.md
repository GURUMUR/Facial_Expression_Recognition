# Facial_Expression_Recognition

## Description :
This code implements real-time Facial Expression Recognition using a pre-trained convolutional neural network (CNN). It utilizes OpenCV for face detection, extracts facial features, and feeds them into the CNN model to predict emotions such as anger, disgust, fear, happiness, neutral, sad, and surprise. The predicted emotion labels are overlaid on the webcam feed, providing live feedback on detected emotions.

## Team Members:
### ● Gurumurthy S,
### ● Haravasu S

## Pre-Requirements:

● Installed Python environment with OpenCV and Keras libraries.<br>
● Trained model files ('facialemotionmodel.json' and 'facialemotionmodel.h5') available.<br>
● Access to a webcam for real-time video input.<br>

## Project Overview:

This code is a Python script for real-time facial emotion recognition using a pre-trained convolutional neural network (CNN) model.<br>
Here's a brief explanation of each part:<br>

### Imports:

● cv2: OpenCV library for computer vision tasks.<br>
● model_from_json from Keras: to load the trained model architecture from a JSON file.<br>
● numpy: for numerical operations.<br>
  
### Loading the Model:

● The script loads a pre-trained CNN model architecture from a JSON file (facialemotionmodel.json) and its weights from an HDF5 file (facialemotionmodel.h5).
```python
with open("facialemotionmodel.json", "r") as json_file:
    model_json = json_file.read()
    model = model_from_json(model_json)
```

### Haar Cascade Classifier:

● It loads the Haar cascade classifier for detecting faces from OpenCV's data.
```python
haar_file=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
```

### Feature Extraction Function:

● extract_features() is a function to preprocess the input image before feeding it into the neural network.

```python
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1,48,48,1)
    return feature/255.0
```

### Accessing Webcam:

● It accesses the default webcam (VideoCapture(0)).
```python
webcam=cv2.VideoCapture(0)
```

### Labels:

● A dictionary labels is defined to map the output of the model to human-readable emotion labels.
```python
labels = {0 : 'angry', 1 : 'disgust', 2 : 'fear', 3 : 'happy', 4 : 'neutral', 5 : 'sad', 6 : 'surprise'}
```

### Main Loop:

● A while loop captures frames from the webcam continuously.<br>
● Inside the loop, it converts the captured frame to grayscale and detects faces using the Haar cascade classifier.<br>
```python
gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
```
● For each detected face, it extracts the face region, resizes it to the required input size for the model (48x48 pixels), preprocesses it, and then passes it through the model.<br>
```python
image = cv2.resize(image,(48,48))
```
● The predicted emotion label is obtained by finding the maximum probability output from the model's prediction.<br>
```pyton
pred = model.predict(img)
```
● The predicted emotion label is then overlaid on the original frame using OpenCV's putText function.<br>
● The processed frame with the predicted emotion label is displayed using imshow.<br>
● The loop continues until the user presses the 'c' key.<br>
```python
if cv2.waitKey(1) & 0xFF==ord("c"):
    break
```
Overall, this script continuously captures video frames from the webcam, detects faces in each frame, predicts the emotion associated with each detected face using the pre-trained CNN model, and overlays the predicted emotion label on the video feed in real-time.

## Output Images:
<img src="https://github.com/GURUMUR/Facial_Expression_Recognition/assets/144895197/8bf72c62-0ed2-4204-ab82-e4a839db2e3b" width="100" height = "100">
![output_2](https://github.com/GURUMUR/Facial_Expression_Recognition/assets/144895197/108df73f-1c6f-44be-820d-05308dccd8e0)
![output_1](https://github.com/GURUMUR/Facial_Expression_Recognition/assets/144895197/0e9371f8-16f4-43bb-88bf-6acab583e169)
![output_6](https://github.com/GURUMUR/Facial_Expression_Recognition/assets/144895197/cfbff3fb-f4bb-4042-9b16-47ae16d87ca7)

