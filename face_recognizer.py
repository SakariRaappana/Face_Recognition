
""" This is a facial recognition application / script written by Sakari Raappana.
This project is inspired by Andrew Ng on his "Deep learning specialization" course.

As a facial feature extractor I'm using a neural network which is build on top of the ResNet architecture
and trained with 1M face images using a triplet loss function.
The pre-trained model I am using is trained by Hiroki Taniai and being published on github.
https://github.com/nyoki-mtl/keras-facenet
 
The program works so that you put images of faces with the names of the image files being the names of the persons
in to a folder. Then the program constructs 128 dimensional vectors of those images being the "persons with credentials".
In the main loop the web camera is captured and then we use Dlib's frontal face detector to identify any faces in the image
and after face is found it is feeded to the facenet which returns 128 dimensional vector which's cosine similarity is
calculated with all the "persons with credentials" vectors. If the prediction treshold is exceeded then we say that
the person is identified (there are some filtering included).

There are also some hyperparameters that can be tuned to get right balance between being easy to get some prediction and
the prediction to be reliable. """

import cv2
import dlib
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from collections import Counter


model = tf.keras.models.load_model('path_to_the_model/facenet_keras.h5')
image_folder_path = 'path_to_image_folder/images/'
face_detector = dlib.get_frontal_face_detector()


#------------------- Define Functions ------------------

def Get_Credentials():
    face_paths = {}
    for filename in os.listdir(image_folder_path):
        name = os.path.splitext(filename)[0]
        face_paths[name] = image_folder_path + filename

    credential_face_vectors = {}
    for name in face_paths.keys():
        face_path = face_paths[name]
        face_image = cv2.imread(face_path)
        face_vector = Predict_Face(face_image)
        credential_face_vectors[name] = face_vector
    
    return credential_face_vectors

def Read_Web_Cam():
    _, image = cap.read()
    height , width , layers =  image.shape
    new_h=height/2
    new_w=width/2
    image = cv2.resize(image, (int(new_w), int(new_h)))
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return image, image_gray

def Predict_Face(roi):
    if np.any(roi):
        roi_scaled = roi / 255
        resized_roi = cv2.resize(roi_scaled, (160,160))
        tensor = resized_roi.reshape(1, 160, 160, 3)
        return model.predict(tensor)
    else:
        pass

def Draw_Recognized_Graphics(image, x,y,w,h, cosine_avg, person, prediction_history_list):
 
    color = (255, 255, 255)
    color2 = (20, 20, 20)
    stroke = 1
    width = int(x + w)
    height = int(y + h)
    cv2.rectangle(image, (x, y), (w, h), color, stroke)
    
    if len(prediction_history_list) < pred_hist_max_len:
        cv2.rectangle(image, (w, y), (w+275, y+75), color, -1)    
        cv2.putText(image, 'Analyzing...', (w+5, y+20),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (color2), 1)
    else:
        cv2.rectangle(image, (w, y), (w+275, y+75), color, -1)    
        cv2.putText(image, str(person), (w+5, y+20),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (color2), 1)
        cv2.putText(image, 'Cosine similarity:', (w+5, y+45),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (color2), 1)
        cv2.putText(image, str(round(cosine_avg[0], 2)), (w+5, y+68),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (color2), 1)

def Draw_Warning_Graphics(image):
    color = (255, 255, 255)
    color2 = (20, 20, 20)
    cv2.rectangle(image, (0, 0), (600, 100), color, -1) 
    cv2.putText(image, 'More than one person detected..', (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (color2), 1)

def moving_average(cordinate_list, cordinate):
    cordinate_list.append(cordinate)
    if len(cordinate_list) > smoothness_value:
        cordinate_list.pop(0)
    m = len(cordinate_list)
    cumulative_average = np.cumsum(cordinate_list, dtype=float)
    return cumulative_average[m - 1:] / m

def Average_Prediction(prediction_history_list):
    if len(prediction_history_list) > pred_hist_max_len:
        prediction_history_list.pop(0)
        
    name_dict = Counter(prediction_history_list)
    most_likely = max(name_dict.values())

    if most_likely > certainty_threshold:
        return max(name_dict, key=name_dict.get)
    else:
        return 'No Credentials'

#-------------------------------------------------------


#------------------- Hyper Parameters ------------------
prediction_threshold = 0.5
certainty_threshold = 10
pred_hist_max_len = 15
smoothness_value = 3
#-------------------------------------------------------


#------------------- Initialize some lists -------------
prediction_history_list = []
cosine_list = []
x_list = []
y_list = []
w_list = []
h_list = []
#--------------------------------------------------------


#------------------- Main Loop --------------------------
credential_face_vectors = Get_Credentials()
cap = cv2.VideoCapture(0)

while(True):
    
    image, image_gray = Read_Web_Cam()
    faces = face_detector(image_gray)

    if len(faces) < 1:
        prediction_history_list = []

    elif len(faces) > 1:
        Draw_Warning_Graphics(image)
        cv2.imshow('frame', image)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
        continue

    else:  
        for i, d in enumerate(faces):
            detection, left, top, right, bottom = i, d.left() -10, d.top() -35, d.right() +10, d.bottom() +10        
    
            roi = image[top:bottom, left:right]
            prediction_vector = Predict_Face(roi)

            for name in credential_face_vectors.keys():
                cosine = cosine_similarity(credential_face_vectors[name], prediction_vector)
                predicted_person = 'No Credentials'
                if cosine > prediction_threshold:
                    predicted_person = name
                    break
                
            prediction_history_list.append(predicted_person)
            person = Average_Prediction(prediction_history_list)

            cosine_avg = moving_average(cosine_list, cosine)

            x_rect = moving_average(x_list, left)
            y_rect = moving_average(y_list, top)
            w_rect = moving_average(w_list, right)
            h_rect = moving_average(h_list, bottom)

            Draw_Recognized_Graphics(image, x_rect, y_rect, w_rect, h_rect, cosine_avg, person, prediction_history_list)

    cv2.imshow('frame', image)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyWindow('frame')