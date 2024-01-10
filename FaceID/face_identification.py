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
        result_image = input_image.copy()
        for result in results :
            for person in face_bank:
                facebank_person_embedding = person["embedding"]
                print(facebank_person_embedding)
                new_person_embedding = result["embedding"]
                print(new_person_embedding)
                distance = np.sqrt(np.sum((facebank_person_embedding - new_person_embedding)**2))
                print("distance value : " , distance)
                if distance < 26 :
                    cv2.rectangle(result_image , (int(result.bbox[0])-10 , int(result.bbox[1])-20 , int(result.bbox[2])-140 , int(result.bbox[3])-25) , (0 , 255 , 255) , 2)
                    cv2.putText(result_image , person["name"] , (int(result.bbox[0]) + 10 , int(result.bbox[1]) -30 ) , 
                    cv2.FONT_HERSHEY_SIMPLEX , 0.8  , (0 , 255, 255) ,  2 , cv2.LINE_AA )
                    print( person["name"])
                    print("true")
                    return True
                else :
                    return False
                break
            else :
                print("False")
                return False
                
    def update_facebank(self ,app , folderpath):
        CreateFaceBank.facebank(self , app=app ,folder_path=folderpath)


