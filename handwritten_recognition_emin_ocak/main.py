import cv2
import pickle
import matplotlib.pyplot as plt

path = r""

class Model:
    def __init__(self):
        self.model = pickle.load(open('read_hand_write' , 'rb'))
        print(f"\nModel is ready.\n")

    def read_to_image(self , path):
        img = cv2.imread(path , cv2.IMREAD_GRAYSCALE)
        img = img.flatten()
        return img

HandwrittenModel = Model()
read_to_image = HandwrittenModel.read_to_image

model = HandwrittenModel.model

img = read_to_image(path)

p = model.predict([img])
print(f"\nResult: {p}\n")