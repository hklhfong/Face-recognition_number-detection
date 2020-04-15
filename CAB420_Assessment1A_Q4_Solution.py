import os
import glob
import cv2

path = r'C:\Users\user\Downloads\CAB420_Assessment1A_Data\Data\Q4\UTKFace\*.jpg'
files = glob.glob(path)

data = []
for f in files:
    d = {}
    head, tail = os.path.split(f)
    parts = tail.split('_')
    if (len(parts) == 4):
        d['age'] = int(parts[0])
        d['gender'] = int(parts[1])
        d['race'] = int(parts[2])
        d['image'] = cv2.imread(f)
        data.append(d)
    else:
        print('Could not load: ' + f + '! Incorrectly formatted filename')
        


    
    
    