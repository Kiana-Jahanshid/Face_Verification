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
    
    def load_image(self , args):
        input_image = cv2.imread(args.image)
        input_image = cv2.cvtColor(input_image , cv2.COLOR_BGR2RGB)
        plt.imshow(input_image)
        return input_image
    
    def load_facebank(self , input_image , app):
        results = app.get(input_image )
        face_bank = np.load("face_bank.npy" , allow_pickle=True  )
        return results , face_bank
    

    def identification(self ,input_image , results , face_bank):
        result_image = input_image.copy()
        counter =0
        for result in results :
            for person in face_bank:
                facebank_person_embedding = person["embedding"]
                new_person_embedding = result["embedding"]
                distance = np.sqrt(np.sum((facebank_person_embedding - new_person_embedding)**2))
                if distance < 25 :
                    if counter <= 1 :
                        cv2.rectangle(result_image , (int(result.bbox[0])-10 , int(result.bbox[1])-20 , int(result.bbox[2])-250 , int(result.bbox[3])-25) , (0 , 255 , 255) , 2)
                        cv2.putText(result_image , person["name"] , (int(result.bbox[0]) + 10 , int(result.bbox[1]) -30 ) , 
                        cv2.FONT_HERSHEY_SIMPLEX , 0.8  , (0 , 255, 255) ,  2 , cv2.LINE_AA )
                        print( person["name"])
                    
                    else :
                        cv2.rectangle(result_image , (int(result.bbox[0])-10 , int(result.bbox[1])-20 , int(result.bbox[2])-140 , int(result.bbox[3])-25) , (0 , 255 , 255) , 2)
                        cv2.putText(result_image , person["name"] , (int(result.bbox[0]) + 10 , int(result.bbox[1]) -30 ) , 
                        cv2.FONT_HERSHEY_SIMPLEX , 0.8  , (0 , 255, 255) ,  2 , cv2.LINE_AA )
                        print( person["name"])
                    break
            else :
                    cv2.rectangle(result_image , (int(result.bbox[0])-10 , int(result.bbox[1])-20 , int(result.bbox[2])-15 , int(result.bbox[3])-20) , (0 , 255 , 255) , 2)
                    cv2.putText(result_image , "unknown" , (int(result.bbox[0]) - 20 , int(result.bbox[1]) -30 ) , 
                        cv2.FONT_HERSHEY_SIMPLEX , 0.8  , (255 , 255, 255) ,  2 , cv2.LINE_AA )
            counter+=1
        plt.imshow(result_image)
        plt.show()


    def update_facebank(self ,app , args):
        CreateFaceBank.facebank(self , app=app ,folder_path=args.update)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--image" , type=str , required=True , help="select an input image")
    parser.add_argument("--update" , help="update face_bank.npy")
    args = parser.parse_args()
    obj = FaceIdentification()
    app = obj.load_model()
    obj.update_facebank(app , args)
    image = obj.load_image(args)
    results , face_bank = obj.load_facebank(input_image=image , app=app)
    obj.identification(input_image=image , results=results , face_bank=face_bank)


#python face_identification.py --image "./group_1.JPG" --update "./face_bank/" 
