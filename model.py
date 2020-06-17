import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.python.keras.backend import set_session
import numpy as np

#Lets configure VRAM allowed for GPU
config = tf.compat.v1.ConfigProto() #Allows config editing for GPU
config.gpu_options.per_process_gpu_memory_fraction = 0.15
session = tf.compat.v1.Session(config = config)
set_session(session)

#Create a FacialExpression class
class FacialExpressionModel(object):
    #Create list for class labels
    Emotions_List = ["Angry", "Fear", "Happy", "Suprise", "Sad", "Neutral", "Disgust"]

    #def init method that takes json file and weights
    def __init__(self, model_json_file, model_weights_file):
        with open(model_json_file, "r") as json_file: #Read the file as json_file
            loaded_model_json = json_file.read() #loaded model is json file
            self.loaded_model = model_from_json(loaded_model_json)

        self.loaded_model.load_weights(model_weights_file) #loaded weights from file for the loade model
        #self.loaded_model.compile()
        #self.loaded_model._make_predict_function() #make a prediction using loaded_model

    def predict_emotion(self, img): #predict function to predict image
        global session
        set_session(session)
        self.preds = self.loaded_model.predict(img)
        
        #We use argmax function to return the class with highest probability. (Largest +ve value)
        return FacialExpressionModel.Emotions_List[np.argmax(self.preds)]#Prediction class is the highest probability class in Emotions_List


