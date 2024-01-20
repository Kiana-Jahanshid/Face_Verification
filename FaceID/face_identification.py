import os
import cv2
import argparse
import numpy as np 
import matplotlib.pyplot as plt 
from insightface.app import FaceAnalysis
from create_facebank import CreateFaceBank


class FaceIdentification :
    def __init__(self):
        pass

    def load_model(self):
        app = FaceAnalysis(name="buffalo_s" , providers=["CPUExecutionProvider"] )
        app.prepare(ctx_id=0 , det_size=(640,640) )
        return app
    
    def load_image(self , image_path):
        input_image = cv2.imread(image_path)
        input_image = cv2.cvtColor(input_image , cv2.COLOR_BGR2RGB)
        plt.imshow(input_image)
        return input_image
    
    def load_facebank(self , input_image , app):
        results = app.get(input_image)
        face_bank = np.load("face_bank.npy" , allow_pickle=True  )
        return results , face_bank
    

    def identification(self ,input_image , results , face_bank):
        for result in results:
            for person in face_bank:
                facebank_person_embedding = person["embedding"]
                new_person_embedding = result["embedding"]

                distance = np.sqrt(np.sum((facebank_person_embedding - new_person_embedding) **2))
                print(distance)
                if distance < 29:
                    cv2.rectangle(input_image,(int(result.bbox[0]),int(result.bbox[1])),(int(result.bbox[2]),int(result.bbox[3])),(0,255,0),2)
                    cv2.putText(input_image,person["name"],(int(result.bbox[0]) , int(result.bbox[1])-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)
                    accept_playing = True
                    temp = False
                    return  accept_playing ,  temp
                    break
            else:
                cv2.rectangle(input_image,(int(result.bbox[0]),int(result.bbox[1])),(int(result.bbox[2]),int(result.bbox[3])),(0,255,0),2)
                cv2.putText(input_image,"Unknown",(int(result.bbox[0]) , int(result.bbox[1])-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)
                accept_playing = False
                temp = False
                return accept_playing , temp
            
    def update_facebank(self ,app , folderpath):
        CreateFaceBank.facebank(self , app=app ,folder_path=folderpath)


