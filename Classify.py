import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from keras.models import load_model

model = load_model('surgical_tools_model.h5')

def open_image():
    file_path = filedialog.askopenfilename()
    img = Image.open(file_path)
    img = img.resize((300, 300)) 
    img = ImageTk.PhotoImage(img)
    label.config(image=img)
    label.image = img 
    predict_label(file_path)

def predict_label(image_path):
    img = Image.open(image_path)
    img = img.resize((150, 150)) 
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array) 
    instrument = np.argmax(prediction)
    confidence = np.max(prediction) 
    if instrument == 0:
        instrument_label.config(text="Prediction: Injection Syringe\nConfidence: {:.2%}".format(confidence))
    elif instrument == 1:
        instrument_label.config(text="Prediction: Mayo Scissors\nConfidence: {:.2%}".format(confidence))
    elif instrument == 2:
        instrument_label.config(text="Prediction: Tissue Forceps\nConfidence: {:.2%}".format(confidence))

root = tk.Tk()
root.title("Surgical Instruments Classifier")
root.geometry("500x500")

label = tk.Label(root)
label.pack(pady=10)

instrument_label = tk.Label(root, text="", font=("Helvetica", 16), wraplength=400)
instrument_label.pack(pady=10)

open_button = tk.Button(root, text="Open Image", command=open_image)
open_button.pack(pady=10)

root.mainloop()
