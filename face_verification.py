import cv2
import argparse
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image



def faceanalysis(opt) :

    app = FaceAnalysis(name="buffalo_s" , providers=['CUDAExecutionProvider' ,'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    # used SCRFD algorithm instead of YOLO

    img = cv2.imread(opt.image_1)
    faces_result = app.get(img)  #prediction
    print(faces_result)
    rimg = app.draw_on(img, faces_result)
    cv2.imwrite("output1.jpg", rimg)
    embedding_1 = faces_result[0]["embedding"] # feature vector first image
    #print(embedding_1) # (512,)


    img2 = cv2.imread(opt.image_2)
    faces_result2 = app.get(img2)  #prediction
    print(faces_result2)
    rimg2 = app.draw_on(img2, faces_result2)
    cv2.imwrite("output2.jpg", rimg2)
    embedding_2 = faces_result2[0]["embedding"] # feature vector second image
    #print(embedding_2) # (512,) == 512D


    euclidean_distance = np.sqrt(np.sum((embedding_1 - embedding_2)**2)) 
    # اگر فرض بر این باشد که ابعاد همیشه فیچر وکتور ۵۱۲ بعدی است دیگه نیازی نیست تقسیم بر ۵۱۲ بکنیم سام را
    # و دیگه نیاز به اسکیلینگ نیست
    print("\neuclidean distance value = " , euclidean_distance)
    # same persons have least euclidean distance between their feature vectors 
    if euclidean_distance < 25 :
        print(" Same Person ")
    elif euclidean_distance > 29 :
        print(" Different Persons ")



if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_1" , type=str , required=True )
    parser.add_argument("--image_2" , type=str , required=True )
    opt = parser.parse_args()
    faceanalysis(opt)


#python face_verification.py --image_1 "./me1.jpg" --image_2 "./me2.jpg"