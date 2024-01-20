import os
import cv2
import numpy as np 
import matplotlib.pyplot as plt 
from insightface.app import FaceAnalysis

class CreateFaceBank:
    def __init__(self):
        pass

    def load_model(self):
        app = FaceAnalysis(name="buffalo_s" , providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0 , det_size=(640,640))
        return app

    def facebank(self , app , folder_path):
        facebank_path = folder_path #"./face_bank/"
        face_bank_embeddings = []
        for person_folder_name in os.listdir(facebank_path):
            folder_path = os.path.join(facebank_path , person_folder_name)
            #print(folder_path)
            if os.path.isdir(folder_path): # if it was a folder 
                for image_name in os.listdir(folder_path):
                    image_path = os.path.join(folder_path , image_name)# complete path
                    print(image_path)
                    image = cv2.imread(image_path)
                    #image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
                    result = app.get(image) # it's a list of persons in an image  
                    if len(result) > 1 : # it means that if more than one person exists in this image
                        print("warning :  more than one face detected in image ")
                        continue # ignore images with more than one person
                    embedding = result[0]["embedding"] #an array of length 512 
                    my_dict = {"name": person_folder_name , "embedding":embedding}
                    face_bank_embeddings.append(my_dict)

        #print(face_bank_embeddings)
        np.save("face_bank.npy" , face_bank_embeddings)