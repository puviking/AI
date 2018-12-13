
# coding: utf-8

# In[2]:


from skimage import measure
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pyaudio  
import wave 


# In[3]:


cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('/Users/Administrator/Documents/Machine Learning/CV2/data/haarcascade_frontalface_alt2.xml')
img1=cv2.imread('/Users/Administrator/Documents/Machine Learning/CV2/Ashwin.png')
img1 = cv2.resize(img1, (200,200))

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
                       
              
        color = (255, 0, 0)
        stroke = 2
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
    
        img2 = roi_color
        img2 = cv2.resize(img2, (200,200))
        ssim = measure.compare_ssim(img1,img2,multichannel='true')
        print("SSIM===",ssim)
        
        if float(ssim) > 0.8:
            chunk = 1024  
            f = wave.open(r"/Users/Administrator/Documents/Machine Learning/CV2/Ashwin.wav","rb")  
            p = pyaudio.PyAudio()
            stream = p.open(format = p.get_format_from_width(f.getsampwidth()),  
                        channels = f.getnchannels(),  
                        rate = f.getframerate(),  
                        output = True)
            data = f.readframes(chunk) 
            while data:  
                stream.write(data)  
                data = f.readframes(chunk)
            stream.stop_stream()  
            stream.close() 
            p.terminate() 
                         
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        img_item = "Test.png"
        cv2.imwrite('/Users/Administrator/Documents/Machine Learning/CV2/'+img_item, roi_color)
        break

# When everything done, release the capture

cap.release()
cv2.destroyAllWindows()

