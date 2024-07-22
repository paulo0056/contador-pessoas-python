import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import*
import cvzone
import numpy as np

#Usando o modelo YOLO para reconhecimento de objetos
model=YOLO('yolov8n.pt')

#Esse codigo é usando a camera mais perto da porta 

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        point = [x, y]
        print(point)
  
        

#Aqui é para abrir a camera
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
cap=cv2.VideoCapture('video1.mp4')

#Leitura do arquivo com os tipos de objetos
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 


count=0


#Importação do tracker, arquivo responsavel para o rastreio dos objetos no video
tracker = Tracker()

#area1 = [(190, 270), (460, 370), (460, 420), (190, 320)]

#Desenho das areas para marcação e contagem
area1 = [(220, 270), (680, 370), (680, 420), (220, 320)]
area2=[(220, 350), (680, 450), (680, 490), (220, 390)]
going_out={}
going_in={}
counter1=[]
counter2=[]

#Aqui em baixo é a logica para a criação dos retangulos, identificação e contagem
while True:    
    ret,frame = cap.read()
    if not ret:
        break


#    count += 1
#    if count % 3 != 0:
#        continue
    frame=cv2.resize(frame,(1020,500))
   

    results=model.predict(frame)
 #   print(results)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
#    print(px)
    
    list=[]
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        
        c=class_list[d]
        if 'person' in c:
            list.append([x1,y1,x2,y2])
    bbox_idx=tracker.update(list)
    for bbox in bbox_idx:
        x3,y3,x4,y4,id=bbox
        result=cv2.pointPolygonTest(np.array(area2,np.int32), ((x4,y4)),False)
        cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 255, 255), 2)
        cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 1)
        
        if result>=0:
            going_out[id]=(x4,y4)
        if id in going_out:
             result1=cv2.pointPolygonTest(np.array(area1,np.int32), ((x4,y4)),False)
             if result1>=0:
                #Cria os circulos e retangulos, colocando também a contagem
                cv2.circle(frame,(x4,y4),7,(255,0,255),-1)    
                cv2.rectangle(frame, (x3,y3), (x4, y4), (255,255,255), 2)
                cvzone.putTextRect(frame,f'{id}',(x3,y3),1,1)
                if counter1.count(id)==0:
                   counter1.append(id)

        result2=cv2.pointPolygonTest(np.array(area1,np.int32), ((x4,y4)),False)
        if result2>=0:
            going_in[id]=(x4,y4)
        if id in going_in:
             result3=cv2.pointPolygonTest(np.array(area2,np.int32), ((x4,y4)),False)
             if result3>=0:
                cv2.circle(frame,(x4,y4),7,(255,0,255),-1)    
                cv2.rectangle(frame, (x3,y3), (x4, y4), (255,255,255), 2)
                cvzone.putTextRect(frame,f'{id}',(x3,y3),1,1)
                if counter2.count(id)==0:
                   counter2.append(id)


        

    out_c=(len(counter1))
    inc_c=(len(counter2))
    cvzone.putTextRect(frame,f'SAIU: {out_c}',(50,60),2,2)
    cvzone.putTextRect(frame,f'ENTROU: {inc_c}',(50,160),2,2)
    cv2.polylines(frame, [np.array(area1,np.int32)], True, (0,255,0),1)
    cv2.polylines(frame, [np.array(area2,np.int32)], True, (0,255,0),1)
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()

