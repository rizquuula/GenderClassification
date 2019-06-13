import os
import cv2
import tensorflow as tf 

gender = ['Perempuan','Laki-laki']
directory = 'Testing/'    
isi = os.listdir(directory)

def testingIMG(filepath):
    size = 128
    img_array = cv2.imread(filepath)#, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (size,size))
    return new_array.reshape(-1,size,size,3)

model = tf.keras.models.load_model("kornet.model")

#prediction = model.predict([testingIMG(test_data_dir)])
#print(prediction)

for i in isi:
    
    data = str(directory+i)
    prediction = model.predict([testingIMG(data)])
    print(data,' = ',gender[int(prediction[0][0])])