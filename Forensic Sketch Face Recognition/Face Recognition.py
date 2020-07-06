
"""
import os
import tkinter as tk
from tkinter import *
import sqlite3
window = tk.Tk()
window.title("Face_Recogniser")
window.geometry("1500x700")
photo = PhotoImage(file='BG1.png')
label = Label(window,image=photo)

def collect_dataset():

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    camera = cv2.VideoCapture(1)
    total_images_counter = 0

    while True:
        img_flag,img_frame = camera.read()
        face = face_cascade.detectMultiScale(img_frame,1.3,5)

        for start_x,start_y,width_measure,height_measure in face:
            total_images_counter+=1
            ROI_crop_face = img_frame[start_y:start_y+height_measure,start_x:start_x+width_measure]
            cv2.imwrite('mydataset/Users.'+str(ID.get())+'.'+str(total_images_counter)+'.jpg',ROI_crop_face)
        cv2.imshow('Collect DataSet',img_frame)

        if cv2.waitKey(1)==ord('q') or total_images_counter>20:
            break

    camera.release()
    cv2.destroyAllWindows()
    return

def entry_Data(user_id,user_name):

    detection = collect_dataset()
    db = sqlite3.connect("Final_DB.db")
    cur = db.cursor()
    query = cur.execute('INSERT INTO Record VALUES (?,?);', (user_id, user_name))
    print("Entry Added To Database")
    db.commit()
    showinfo(title="Data Insert Notification", message="Data inserted To table")
    return

load_rgb_data_set = os.listdir("C:/Users/Muhammad Bilal Zafar/PycharmProjects/Face_Recognition_LBPH/mydataset")
#print(ging)
#sampleNum = 0

def GrayScale(RGB_img):

    r = RGB_img[:, :, 2]
    g = RGB_img[:, :, 2]
    b = RGB_img[:, :, 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray

def Sketch(front_blur, back_invert):

    result = front_blur * 255 / (255 - back_invert)
    result[result > 255] = 255
    result[back_invert == 255] = 255
    return result.astype('uint8')

def sketch_Conversion():

    showinfo(title="Sketch Conversion Notification", message="Conversion Start!!!! Wait for a moment")
    sampleNum = 0

    for rgb_data_set_element in load_rgb_data_set:
        sampleNum = sampleNum + 1
        img = cv2.imread("C:/Users/Muhammad Bilal Zafar/PycharmProjects/Face_Recognition_LBPH/mydataset/" + rgb_data_set_element)
        #srcImgRZ = cv2.resize(img, (400, 400))
        img_gray = GrayScale(img)
        inverted_img = 255 - img_gray
        blur_img = cv2.GaussianBlur(inverted_img, ksize=(221, 221), sigmaX=75, sigmaY=75)
        sketch_img = Sketch(blur_img, img_gray)
        cv2.imwrite("mydatasetconverted/" + rgb_data_set_element, sketch_img)
        #cv2.imshow('Image STEP IV', final_img)
        cv2.waitKey(100)
        cv2.destroyAllWindows()
    print('Successfully Conversion Done!!!!!')
    showinfo(title="Sketch Conversion Notification", message="Congragulations!!! Sketch Conversion Successfully")


#--------------------------------------------------------------------------------

import tkinter as tk
from tkinter.messagebox import showinfo
import cv2, os
from tkinter import filedialog
import numpy as np
from PIL import Image, ImageTk

def train_all_Images():

    showinfo(title="Trainnig Notification", message="Training Start!!!! Wait for a moment")
    recognizer_model = cv2.face_LBPHFaceRecognizer.create()  # recognizer = cv2.face.LBPHFaceRecognizer_create()#$cv2.createLBPHFaceRecognizer()
    face_file_path = "haarcascade_frontalface_default.xml"
    cascade_detector = cv2.CascadeClassifier(face_file_path)
    all_faces, all_Id = get_all_Images_with_labels("mydatasetconverted")
    recognizer_model.train(all_faces, np.array(all_Id))
    recognizer_model.save("recognizer2/trainningData2.yml")
    showinfo(title="Data Notification", message="Congragulations!!! Data Trained")

def get_all_Images_with_labels(path):

    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    images_list = []
    Ids_list = []

    for image_paths_element in image_paths:
        images_with_pil = Image.open(image_paths_element).convert('L')
        numpy_images = np.array(images_with_pil, 'uint8')
        Id = int(os.path.split(image_paths_element)[-1].split(".")[1])
        images_list.append(numpy_images)
        Ids_list.append(Id)
    return images_list, Ids_list

#----------------------------------------------------------------------------------

model_to_recognize = cv2.face.LBPHFaceRecognizer_create()
model_to_recognize.read("recognizer/trainningData.yml")
#path = 'dataSet'

id = 0
def get_profile_with_data(id):

    conn = sqlite3.connect("Final_DB_1.db")
    query="SELECT * FROM Record WHERE ID="+str(id)
    cursor_obj = conn.execute(query)
    profile_data = None

    for row in cursor_obj:
        profile_data = row
    conn.close()
    return profile_data

def image_browsing():

    browsed_image = filedialog.askopenfilename()  # asking user to load an image
    cam = cv2.imread(browsed_image)
    resize = cv2.resize(cam,(320,320))
    gray = cv2.cvtColor(resize,cv2.COLOR_BGR2GRAY)
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    detected_face = face_detector.detectMultiScale(gray,1.3,5)

    for (initial_x,initial_y,width,height) in detected_face:
        id, threshold_value = model_to_recognize.predict(gray[initial_y:initial_y + height, initial_x:initial_x + width])
        print(threshold_value)

        def thresholding(threshold_level):

            threshold_level = int(75*(threshold_level/100))
            return threshold_level

        threshold_result = thresholding(threshold_value)
        cv2.rectangle(resize,(initial_x,initial_y),(initial_x+width,initial_y+height),(0,0,255),2)

        if threshold_result >= 70:
            profile = get_profile_with_data(id)

        else:
            profile = 'No Profile'

        print(profile)
        print(threshold_result)
    #    if profile != None:
#        Age = str(profile[2])
        if threshold_result >= 70:
#            cv2.putText(resize,str("Name: "+profile[1]),(initial_x,initial_y+height+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
#            cv2.putText(resize,str("Age: "+Age),(initial_x,initial_y+height+60),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
#            cv2.putText(resize,str("Crime: "+profile[3]),(initial_x,initial_y+height+90),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
            cv2.putText(resize,str(profile[1]),(initial_x,initial_y+height+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
            cv2.putText(resize,str(profile[2]),(initial_x,initial_y+height+60),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
            cv2.putText(resize,str(profile[3]),(initial_x,initial_y+height+90),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)

    #       cv2.putText(resize,str(profile[3]), (x, y + h+90), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
    #        cv2.putText(resize,str(profile[4]), (x, y + h+120), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

        else:
    #        print('not found')
            cv2.putText(resize,str('UnWanted'),(initial_x,initial_y+height+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
        cv2.imshow("Faces",resize)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

collect_Img = tk.Button(window, text="Collect Data", command=lambda: entry_Data(ID.get(),Name.get()), fg="black", bg="white", width=15, height=2,
                    activebackground="Red", font=('times', 15, ' bold '))
collect_Img.place(x=0, y=580)

dataSet_conversion = tk.Button(window, text="Conversion", command=sketch_Conversion, fg="black", bg="white", width=15, height=2,
                    activebackground="Red", font=('times', 15, ' bold '))
dataSet_conversion.place(x=200, y=580)

load_Img = tk.Button(window, text="Browse Image", command=image_browsing, fg="black", bg="white", width=15, height=2,
                   activebackground="white", font=('times', 15, ' bold '))
load_Img.place(x=400, y=580)

train_Img = tk.Button(window, text="Train Images", command=train_all_Images, fg="black", bg="white", width=15, height=2,
                     activebackground="Red", font=('times', 15, ' bold '))
train_Img.place(x=600, y=580)

quitWindow = tk.Button(window, text="Quit", command=window.destroy, fg="black", bg="white", width=15, height=2,
                       activebackground="white", font=('times', 15, ' bold '))
quitWindow.place(x=800, y=580)

label.pack()

id_label = Label(window,text='ID',width=10,bg='Sky Blue',fg='black',font=('times', 15, ' bold '))
ID = IntVar()
#ID = StringVar()
id_label_entry = Entry(window,width=10,textvariable=ID,bg='Sky Blue',font=('times',15,'bold'))
id_label_entry.place(x=0,y=150)
id_label.place(x=0, y=100)

name_label = Label(window,text='NAME',width=10,bg='Sky Blue',fg='black',font=('times', 15, ' bold '))
name_label.pack(pady=10,padx=10)
Name = StringVar()
name_label_entry = Entry(window,textvariable=Name,width=10,bg='Sky Blue',fg='black',font=('times', 15, ' bold '))
name_label.place(x=0, y=200)
name_label_entry.place(x=0,y=250)

age_label = Label(window,text='AGE',width=10,bg='Sky Blue',fg='black',font=('times',15,'bold'))
age_label.pack(padx=10,pady=10)
Age = StringVar()
age_label_enntry = Entry(window,textvariable=Age,width=10,bg='Sky Blue',fg='black',font=('times',15,'bold'))
age_label.place(x=0,y=300)
age_label_enntry.place(x=0,y=350)

criminal_record_label = Label(window,text='Crime Status',width=10,bg='Sky Blue',fg='black',font=('times',15,'bold'))
criminal_record_label.place(x=0,y=400)
Crime = StringVar()
criminal_record_label_entry = Entry(window,textvariable=Crime,width=10,bg='Sky Blue',fg='black',font=('times',15,'bold'))
criminal_record_label_entry.place(x=0,y=450)
#criminal_record_label.pack(padx=10,pady=10)

window.mainloop()

"""


##############################################################################################################


import os
import tkinter as tk
from tkinter import *
import sqlite3
window = tk.Tk()
window.title("Face_Recogniser")
window.geometry("1500x700")
photo = PhotoImage(file='BG1.png')
label = Label(window,image=photo)

def collect_dataset():

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    camera = cv2.VideoCapture(1)
    total_images_counter = 0

    while True:
        img_flag,img_frame = camera.read()
        face = face_cascade.detectMultiScale(img_frame,1.3,5)

        for start_x,start_y,width_measure,height_measure in face:
            total_images_counter+=1
            ROI_crop_face = img_frame[start_y:start_y+height_measure,start_x:start_x+width_measure]
            cv2.imwrite('dataSet/Users.'+str(ID.get())+'.'+str(total_images_counter)+'.jpg',ROI_crop_face)
        cv2.imshow('Collect DataSet',img_frame)

        if cv2.waitKey(1)==ord('q') or total_images_counter>50:
            break

    camera.release()
    cv2.destroyAllWindows()
    return

def entry_Data(user_id,user_name):

    detection = collect_dataset()
    db = sqlite3.connect("myDB.db")
    cur = db.cursor()
    query = cur.execute('INSERT INTO Record VALUES (?,?);', (user_id, user_name))
    print("Entry Added To Database")
    db.commit()
    showinfo(title="Data Insert Notification", message="Data inserted To table")
    return

load_rgb_data_set = os.listdir("C:/Users/MUHAMMAD BILAL ZAFAR/PycharmProjects/Face_Recognition_LBPH/dataSet")
#print(ging)
#sampleNum = 0

def GrayScale(RGB_img):

    r = RGB_img[:, :, 2]
    g = RGB_img[:, :, 2]
    b = RGB_img[:, :, 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray

def Sketch(front_blur, back_invert):

    result = front_blur * 255 / (255 - back_invert)
    result[result > 255] = 255
    result[back_invert == 255] = 255
    return result.astype('uint8')

def sketch_Conversion():

    showinfo(title="Sketch Conversion Notification", message="Conversion Start!!!! Wait for a moment")
    sampleNum = 0

    for rgb_data_set_element in load_rgb_data_set:
        sampleNum = sampleNum + 1
        img = cv2.imread("C:/Users/MUHAMMAD BILAL ZAFAR/PycharmProjects/Face_Recognition_LBPH/dataSet/" + rgb_data_set_element)
        #srcImgRZ = cv2.resize(img, (400, 400))
        img_gray = GrayScale(img)
        inverted_img = 255 - img_gray
        blur_img = cv2.GaussianBlur(inverted_img, ksize=(221, 221), sigmaX=75, sigmaY=75)
        sketch_img = Sketch(blur_img, img_gray)
        cv2.imwrite("dataSetConvertIntoSketch/" + rgb_data_set_element, sketch_img)
        #cv2.imshow('Image STEP IV', final_img)
        cv2.waitKey(10)
        cv2.destroyAllWindows()
    print('Successfully Conversion Done!!!!!')
    showinfo(title="Sketch Conversion Notification", message="Congragulations!!! Sketch Conversion Successfully")


#--------------------------------------------------------------------------------

import tkinter as tk
from tkinter.messagebox import showinfo
import cv2, os
from tkinter import filedialog
import numpy as np
from PIL import Image, ImageTk

def train_all_Images():

    showinfo(title="Trainnig Notification", message="Training Start!!!! Wait for a moment")
    recognizer_model = cv2.face_LBPHFaceRecognizer.create()  # recognizer = cv2.face.LBPHFaceRecognizer_create()#$cv2.createLBPHFaceRecognizer()
    face_file_path = "haarcascade_frontalface_default.xml"
    cascade_detector = cv2.CascadeClassifier(face_file_path)
    all_faces, all_Id = get_all_Images_with_labels("dataSetConvertIntoSketch")
    recognizer_model.train(all_faces, np.array(all_Id))
    recognizer_model.save("recognizer2/trainningData1.yml")
    showinfo(title="Data Notification", message="Congragulations!!! Data Trained")

def get_all_Images_with_labels(path):

    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    images_list = []
    Ids_list = []

    for image_paths_element in image_paths:
        images_with_pil = Image.open(image_paths_element).convert('L')
        numpy_images = np.array(images_with_pil, 'uint8')
        Id = int(os.path.split(image_paths_element)[-1].split(".")[1])
        images_list.append(numpy_images)
        Ids_list.append(Id)
    return images_list, Ids_list

#----------------------------------------------------------------------------------

model_to_recognize = cv2.face.LBPHFaceRecognizer_create()
model_to_recognize.read("recognizer2/trainningData1.yml")
00#path = 'dataSet'

id = 0
def get_profile_with_data(id):

    conn = sqlite3.connect("myDB.db")
    query="SELECT * FROM Record WHERE ID="+str(id)
    cursor_obj = conn.execute(query)
    profile_data = None

    for row in cursor_obj:
        profile_data = row
    conn.close()
    return profile_data

def image_browsing():

    browsed_image = filedialog.askopenfilename()  # asking user to load an image
    cam = cv2.imread(browsed_image)
    resize = cv2.resize(cam,(520,520))
    gray = cv2.cvtColor(resize,cv2.COLOR_BGR2GRAY)
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    detected_face = face_detector.detectMultiScale(gray,1.3,5)

    for (initial_x,initial_y,width,height) in detected_face:
        id, threshold_value = model_to_recognize.predict(gray[initial_y:initial_y + height, initial_x:initial_x + width])
        print(threshold_value)

        def thresholding(threshold_level):

            threshold_level = int(65*(threshold_level/100))
            return threshold_level

        threshold_result = thresholding(threshold_value)
        cv2.rectangle(resize,(initial_x,initial_y),(initial_x+width,initial_y+height),(0,0,255),2)

        if threshold_result >= 70:
            profile = get_profile_with_data(id)

        else:
            profile = 'No Profile'

        print(profile)
        print(threshold_result)
    #    if profile != None:

        if threshold_result >= 70:
            cv2.putText(resize,str(profile[1]),(initial_x,initial_y+height+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
    #        cv2.putText(resize,str(profile[2]), (x, y + h+60), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
    #       cv2.putText(resize,str(profile[3]), (x, y + h+90), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
    #        cv2.putText(resize,str(profile[4]), (x, y + h+120), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

        else:
    #        print('not found')
            cv2.putText(resize,str('UnWanted'),(initial_x,initial_y+height+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
        cv2.imshow("Faces",resize)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

collect_Img = tk.Button(window, text="Collect Data", command=lambda: entry_Data(ID.get(),Name.get()), fg="black", bg="white", width=15, height=2,
                    activebackground="Red", font=('times', 15, ' bold '))
collect_Img.place(x=0, y=580)

dataSet_conversion = tk.Button(window, text="Conversion", command=sketch_Conversion, fg="black", bg="white", width=15, height=2,
                    activebackground="Red", font=('times', 15, ' bold '))
dataSet_conversion.place(x=200, y=580)

load_Img = tk.Button(window, text="Browse Image", command=image_browsing, fg="black", bg="white", width=15, height=2,
                   activebackground="white", font=('times', 15, ' bold '))
load_Img.place(x=400, y=580)

train_Img = tk.Button(window, text="Train Images", command=train_all_Images, fg="black", bg="white", width=15, height=2,
                     activebackground="Red", font=('times', 15, ' bold '))
train_Img.place(x=600, y=580)

quitWindow = tk.Button(window, text="Quit", command=window.destroy, fg="black", bg="white", width=15, height=2,
                       activebackground="white", font=('times', 15, ' bold '))
quitWindow.place(x=800, y=580)

label.pack()

id_label = Label(window,text='ID',width=10,bg='Sky Blue',fg='black',font=('times', 15, ' bold '))
ID = IntVar()
#ID = StringVar()
id_label_entry = Entry(window,width=10,textvariable=ID,bg='Sky Blue',font=('times',15,'bold'))
id_label_entry.place(x=0,y=150)
id_label.place(x=0, y=100)

name_label = Label(window,text='NAME',width=10,bg='Sky Blue',fg='black',font=('times', 15, ' bold '))
name_label.pack(pady=10,padx=10)
Name = StringVar()
name_label_entry = Entry(window,textvariable=Name,width=10,bg='Sky Blue',fg='black',font=('times', 15, ' bold '))
name_label.place(x=0, y=200)
name_label_entry.place(x=0,y=250)
window.mainloop()




















"""

import os
import tkinter as tk
from tkinter import *
import sqlite3
window = tk.Tk()
window.title("Face_Recogniser")
window.geometry("1500x700")
photo = PhotoImage(file='BG1.png')
label = Label(window,image=photo)
#window.configure(background='Yellow')
#window.grid_rowconfigure(0, weight=1)
#window.grid_columnconfigure(0, weight=1)
def faceDetection():
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    camera = cv2.VideoCapture(1)
    sample = 0
    while True:
        red,frame = camera.read()
        face = cascade.detectMultiScale(frame,1.3,5)
        for x,y,w,h in face:
            sample+=1
            crop_face = frame[y:y+h,x:x+w]
            cv2.imwrite('mydataset/Users.'+str(ID.get())+'.'+str(sample)+'.jpg',crop_face)
        cv2.imshow('Videos',frame)
        if cv2.waitKey(1)==ord('q') or sample>20:
            break
    camera.release()
    cv2.destroyAllWindows()
    return
def entry_Data(id,name):
    detection = faceDetection()
    db = sqlite3.connect("myDB.db")
    cur = db.cursor()
    query = cur.execute('INSERT INTO projects VALUES (?,?);', (id, name))
    print("Entry Added To Database")
    db.commit()
    showinfo(title="Librarian Add", message="Data inserted To table")
    return
ging = os.listdir("C:/Users/MUHAMMAD BILAL ZAFAR/PycharmProjects/ChikonEye_a-third-eye-app-which-protects-your-work-from-peepers.-ashraf-minhaj-patch-1/mydataset")
#print(ging)
#sampleNum = 0
def GrayScale(rgb):
    r = rgb[:, :, 2]
    g = rgb[:, :, 2]
    b = rgb[:, :, 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray
def dodge(front, back):
    result = front * 255 / (255 - back)
    result[result > 255] = 255
    result[back == 255] = 255
    return result.astype('uint8')
def sketch_Conversion():
    showinfo(title="Sketch Conversion Notification", message="Conversion Start!!!! Wait for a moment")
    sampleNum = 0
    for ginglement in ging:
        sampleNum = sampleNum + 1
        img = cv2.imread("C:/Users/MUHAMMAD BILAL ZAFAR/PycharmProjects/ChikonEye_a-third-eye-app-which-protects-your-work-from-peepers.-ashraf-minhaj-patch-1/mydataset/" + ginglement)
        #srcImgRZ = cv2.resize(img, (400, 400))
        img_gray = GrayScale(img)
        inverted_img = 255 - img_gray
        blur_img = cv2.GaussianBlur(inverted_img, ksize=(221, 221), sigmaX=75, sigmaY=75)
        final_img = dodge(blur_img, img_gray)
        cv2.imwrite("mydatasetconverted/" + ginglement, final_img)
        #cv2.imshow('Image STEP IV', final_img)
        cv2.waitKey(100)
        cv2.destroyAllWindows()
    print('Successfully Conversion Done!!!!!')
    showinfo(title="Sketch Conversion Notification", message="Congragulations!!! Sketch Conversion Successfully")


#--------------------------------------------------------------------------------

import tkinter as tk
from tkinter.messagebox import showinfo
import cv2, os
from tkinter import filedialog
import numpy as np
from PIL import Image, ImageTk
def TrainImages():
    showinfo(title="Trainnig Notification", message="Training Start!!!! Wait for a moment")
    recognizer = cv2.face_LBPHFaceRecognizer.create()  # recognizer = cv2.face.LBPHFaceRecognizer_create()#$cv2.createLBPHFaceRecognizer()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    faces, Id = getImagesAndLabels("mydatasetconverted")
    recognizer.train(faces, np.array(Id))
    recognizer.save("recognizer2/trainningData2.yml")
    showinfo(title="Librarian Add", message="Congragulations!!! Data Trained")
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    images = []
    Ids = []
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        images.append(imageNp)
        Ids.append(Id)
    return images, Ids
trainImg = tk.Button(window, text="Train Images", command=TrainImages, fg="black", bg="grey", width=10, height=1,
                     activebackground="Red", font=('times', 15, ' bold '))
trainImg.place(x=200, y=500)

#----------------------------------------------------------------------------------

rec = cv2.face.LBPHFaceRecognizer_create()
rec.read("recognizer/trainningData.yml")
#path = 'dataSet'

id = 0
def getProfile(id):
    conn = sqlite3.connect("myDB.db")
    cmd="SELECT * FROM projects WHERE ID="+str(id)
    cursor = conn.execute(cmd)
    profile = None
    for row in cursor:
        profile = row
    conn.close()
    return profile
def browse():
    myfile = filedialog.askopenfilename()  # asking user to load an image
    cam = cv2.imread(myfile)
    resize = cv2.resize(cam,(520,520))
    gray = cv2.cvtColor(resize,cv2.COLOR_BGR2GRAY)
    faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = faceDetect.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        id, conf = rec.predict(gray[y:y + h, x:x + w])
        print(conf)
        def confidence(confi):
            confi = int(75*(confi/100))
            return confi
        confd = confidence (conf)
        cv2.rectangle(resize,(x,y),(x+w,y+h),(0,0,255),2)
        if confd >= 70:
            profile = getProfile(id)
        else:
            profile = 'No Profile'
        print(profile)
        print(confd)
    #    if profile != None:
        if confd >= 70:
            cv2.putText(resize,str(profile[1]),(x,y+h+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
    #        cv2.putText(resize,str(profile[2]), (x, y + h+60), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
    #       cv2.putText(resize,str(profile[3]), (x, y + h+90), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
    #        cv2.putText(resize,str(profile[4]), (x, y + h+120), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        else:
    #        print('not found')
            cv2.putText(resize,str('UnWanted'),(x,y+h+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
        cv2.imshow("Faces",resize)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
takeImg = tk.Button(window, text="Take Images", command=lambda: entry_Data(ID.get(),Name.get()), fg="black", bg="white", width=15, height=2,
                    activebackground="Red", font=('times', 15, ' bold '))
takeImg.place(x=0, y=300)
conversion = tk.Button(window, text="Conversion", command=sketch_Conversion, fg="black", bg="white", width=15, height=2,
                    activebackground="Red", font=('times', 15, ' bold '))
conversion.place(x=0, y=400)
loadImg = tk.Button(window, text="Open Image", command=browse, fg="black", bg="white", width=15, height=2,
                   activebackground="white", font=('times', 15, ' bold '))
loadImg.place(x=0, y=500)
quitWindow = tk.Button(window, text="Quit", command=window.destroy, fg="black", bg="white", width=15, height=2,
                       activebackground="white", font=('times', 15, ' bold '))
quitWindow.place(x=0, y=600)
label.pack()
id_label = Label(window,text='ID',width=10,bg='Sky Blue',fg='black',font=('times', 15, ' bold '))
#ID = IntVar()
ID = StringVar()
id_label_entry = Entry(window,width=10,textvariable=ID,bg='Sky Blue',font=('times',15,'bold'))
id_label_entry.place(x=0,y=150)
id_label.place(x=0, y=100)
name_label = Label(window,text='NAME',width=10,bg='Sky Blue',fg='black',font=('times', 15, ' bold '))
name_label.pack(pady=10,padx=10)
Name = StringVar()
name_label_entry = Entry(window,textvariable=Name,width=10,bg='Sky Blue',fg='black',font=('times', 15, ' bold '))
name_label.place(x=0, y=200)
name_label_entry.place(x=0,y=250)
window.mainloop()



"""











































"""
import os
import tkinter as tk
from tkinter import *
import sqlite3
window = tk.Tk()
window.title("Face_Recogniser")
window.geometry("1500x700")
photo = PhotoImage(file='BG1.png')
label = Label(window,image=photo)
#window.configure(background='Yellow')
window.grid_rowconfigure(0, weight=1) #?
window.grid_columnconfigure(0, weight=1) #?
def faceDetection():
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    camera = cv2.VideoCapture(1)
    sample = 0
    while True:
        red,frame = camera.read()
        face = cascade.detectMultiScale(frame,1.3,5)
        for x,y,w,h in face:
            sample+=1
            crop_face = frame[y:y+h,x:x+w]
            cv2.imwrite('mydataset/Users.'+str(ID.get())+'.'+str(sample)+'.jpg',crop_face)
        cv2.imshow('Videos',frame)
        if cv2.waitKey(1)==ord('q') or sample>20:
            break
    camera.release()
    cv2.destroyAllWindows()
    return
def entry_Data(id,name):
    detection = faceDetection()
    db = sqlite3.connect("myDB.db")
    cur = db.cursor()
    query = cur.execute('INSERT INTO projects VALUES (?,?);', (id, name))
    print("Entry Added To Database")
    db.commit()
    showinfo(title="Librarian Add", message="Data inserted To table")
    return
ging = os.listdir("C:/Users/MUHAMMAD BILAL ZAFAR/PycharmProjects/ChikonEye_a-third-eye-app-which-protects-your-work-from-peepers.-ashraf-minhaj-patch-1/mydataset")
#print(ging)
sampleNum = 0
def GrayScale(rgb):
    r = rgb[:, :, 2]
    g = rgb[:, :, 2]
    b = rgb[:, :, 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray
def dodge(front, back):
    result = front * 255 / (255 - back)
    result[result > 255] = 255
    result[back == 255] = 255
    return result.astype('uint8')
def sketch_Conversion():
    showinfo(title="Sketch Conversion Notification", message="Conversion Start!!!! Wait for a moment")
    sampleNum = 0
    for ginglement in ging:
        sampleNum = sampleNum + 1
        img = cv2.imread("C:/Users/MUHAMMAD BILAL ZAFAR/PycharmProjects/ChikonEye_a-third-eye-app-which-protects-your-work-from-peepers.-ashraf-minhaj-patch-1/mydataset/" + ginglement)
        #srcImgRZ = cv2.resize(img, (400, 400))
        img_gray = GrayScale(img)
        inverted_img = 255 - img_gray
        blur_img = cv2.GaussianBlur(inverted_img, ksize=(221, 221), sigmaX=75, sigmaY=75)
        final_img = dodge(blur_img, img_gray)
        cv2.imwrite("mydatasetconverted/" + ginglement, final_img)
        #cv2.imshow('Image STEP IV', final_img)
        cv2.waitKey(100)
        cv2.destroyAllWindows()
    print('Successfully Conversion Done!!!!!')
    showinfo(title="Sketch Conversion Notification", message="Congragulations!!! Sketch Conversion Successfully")


#--------------------------------------------------------------------------------

import tkinter as tk
from tkinter import Message, Text
from tkinter.messagebox import showinfo
import cv2, os
from tkinter import filedialog
import numpy as np
from PIL import Image, ImageTk
def TrainImages():
    recognizer = cv2.face_LBPHFaceRecognizer.create()  # recognizer = cv2.face.LBPHFaceRecognizer_create()#$cv2.createLBPHFaceRecognizer()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    faces, Id = getImagesAndLabels("mydatasetconverted")
    recognizer.train(faces, np.array(Id))
    recognizer.save("recognizer2/trainningData2.yml")
    showinfo(title="Librarian Add", message="Congragulations!!! Data Trained")
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    images = []
    Ids = []
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        images.append(imageNp)
        Ids.append(Id)
    return images, Ids
trainImg = tk.Button(window, text="Train Images", command=TrainImages, fg="black", bg="white", width=15, height=1,
                     activebackground="Red", font=('times', 15, ' bold '))
trainImg.place(x=600, y=600)

#----------------------------------------------------------------------------------

rec = cv2.face.LBPHFaceRecognizer_create()
rec.read("recognizer/trainningData.yml")
#path = 'dataSet'

id = 0
def getProfile(id):
    conn = sqlite3.connect("myDB.db")
    cmd="SELECT * FROM projects WHERE ID="+str(id)
    cursor = conn.execute(cmd)
    profile = None
    for row in cursor:
        profile = row
    conn.close()
    return profile
def browse():
    myfile = filedialog.askopenfilename()  # asking user to load an image
    cam = cv2.imread(myfile)
    resize = cv2.resize(cam,(520,520))
    gray = cv2.cvtColor(resize,cv2.COLOR_BGR2GRAY)
    faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = faceDetect.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        id, conf = rec.predict(gray[y:y + h, x:x + w])
        print(conf)
        def confidence(confi):
            confi = int(75*(confi/100))
            return confi
        confd = confidence (conf)
        cv2.rectangle(resize,(x,y),(x+w,y+h),(0,0,255),2)
        if confd >= 70:
            profile = getProfile(id)
        else:
            profile = 'No Profile'
        print(profile)
        print(confd)
    #    if profile != None:
        if confd >= 70:
            cv2.putText(resize,str(profile[1]),(x,y+h+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
    #        cv2.putText(resize,str(profile[2]), (x, y + h+60), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
    #       cv2.putText(resize,str(profile[3]), (x, y + h+90), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
    #        cv2.putText(resize,str(profile[4]), (x, y + h+120), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        else:
    #        print('not found')
            cv2.putText(resize,str('UnWanted'),(x,y+h+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
        cv2.imshow("Faces",resize)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
takeImg = tk.Button(window, text="Take Images", command=lambda: entry_Data(ID.get(),Name.get()), fg="black", bg="white", width=15, height=1,
                    activebackground="Red", font=('times', 15, ' bold '))
takeImg.place(x=0, y=600)
conversion = tk.Button(window, text="Conversion", command=sketch_Conversion, fg="black", bg="white", width=15, height=1,
                    activebackground="Red", font=('times', 15, ' bold '))
conversion.place(x=200, y=600)
loadImg = tk.Button(window, text="Open Image", command=browse, fg="black", bg="white", width=15, height=1,
                   activebackground="white", font=('times', 15, ' bold '))
loadImg.place(x=400, y=600)
quitWindow = tk.Button(window, text="Quit", command=window.destroy, fg="black", bg="white", width=15, height=1,
                       activebackground="white", font=('times', 15, ' bold '))
quitWindow.place(x=800, y=600)
label.pack()

lbl = tk.Label(window, text="Auto ID", width=10, height=1, fg="black", bg="grey", font=('times', 15, ' bold '))
lbl.place(x=20, y=100)
txt1 = tk.Entry(window, width=10, bg="white", fg="red", font=('times', 20, ' bold '))
txt1.place(x=200, y=100)

lbl2 = tk.Label(window, text="Enter Name", width=10, fg="black", bg="grey", height=1, font=('times', 15, ' bold '))
lbl2.place(x=20, y=150)

txt2 = tk.Entry(window, width=10, bg="white", fg="red", font=('times', 20, ' bold '))
txt2.place(x=200, y=150)


#id_label = Label(window,text='ID',font=20,bg='Yellow',fg='black')
ID = IntVar()
#id_label_entry = Entry(window,textvariable=ID,bg='Yellow')
#id_label_entry.place(x=0,y=150)
#id_label.place(x=0, y=100)
#name_label = Label(window,text='NAME',font=20,bg='Yellow',fg='black')
#name_label.pack(pady=10,padx=10)
Name = StringVar()
#name_label_entry = Entry(window,textvariable=Name)
#name_label.place(x=0, y=200)
#name_label_entry.place(x=0,y=250)
window.mainloop()
"""





"""
import os
import tkinter as tk
from tkinter import *
import sqlite3
window = tk.Tk()
window.title("Face_Recogniser")
window.geometry("1500x700")
photo = PhotoImage(file='BG1.png')
label = Label(window,image=photo)
#window.configure(background='Yellow')
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)
def faceDetection():
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    camera = cv2.VideoCapture(1)
    sample = 0
    while True:
        red,frame = camera.read()
        face = cascade.detectMultiScale(frame,1.3,5)
        for x,y,w,h in face:
            sample+=1
            crop_face = frame[y:y+h,x:x+w]
            cv2.imwrite('mydataset/Users.'+str(ID.get())+'.'+str(sample)+'.jpg',crop_face)
        cv2.imshow('Videos',frame)
        if cv2.waitKey(1)==ord('q') or sample>20:
            break
    camera.release()
    cv2.destroyAllWindows()
    return
def entry_Data(id,name):
    detection = faceDetection()
    db = sqlite3.connect("myDB.db")
    cur = db.cursor()
    query = cur.execute('INSERT INTO projects VALUES (?,?);', (id, name))
    print("Entry Added To Database")
    db.commit()
    showinfo(title="Librarian Add", message="Data inserted To table")
    return
ging = os.listdir("C:/Users/MUHAMMAD BILAL ZAFAR/PycharmProjects/ChikonEye_a-third-eye-app-which-protects-your-work-from-peepers.-ashraf-minhaj-patch-1/mydataset")
print(ging)
sampleNum = 0
def GrayScale(rgb):
    r = rgb[:, :, 2]
    g = rgb[:, :, 2]
    b = rgb[:, :, 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray
def dodge(front, back):
    result = front * 255 / (255 - back)
    result[result > 255] = 255
    result[back == 255] = 255
    return result.astype('uint8')
def sketch_Conversion():
    showinfo(title="Sketch Conversion Notification", message="Conversion Start!!!! Wait for a moment")
    sampleNum = 0
    for ginglement in ging:
        sampleNum = sampleNum + 1
        img = cv2.imread("C:/Users/MUHAMMAD BILAL ZAFAR/PycharmProjects/ChikonEye_a-third-eye-app-which-protects-your-work-from-peepers.-ashraf-minhaj-patch-1/mydataset/" + ginglement)
        #srcImgRZ = cv2.resize(img, (400, 400))
        img_gray = GrayScale(img)
        inverted_img = 255 - img_gray
        blur_img = cv2.GaussianBlur(inverted_img, ksize=(221, 221), sigmaX=75, sigmaY=75)
        final_img = dodge(blur_img, img_gray)
        cv2.imwrite("mydatasetconverted/" + ginglement, final_img)
        #cv2.imshow('Image STEP IV', final_img)
        cv2.waitKey(100)
        cv2.destroyAllWindows()
    print('Successfully Conversion Done!!!!!')
    showinfo(title="Sketch Conversion Notification", message="Congragulations!!! Sketch Conversion Successfully")


#--------------------------------------------------------------------------------


import tkinter as tk
from tkinter.messagebox import showinfo
import cv2, os
from tkinter import filedialog
import numpy as np
from PIL import Image, ImageTk
def TrainImages():
    showinfo(title="Trainnig Notification", message="Training Start!!!! Wait for a moment")
    recognizer = cv2.face_LBPHFaceRecognizer.create()  # recognizer = cv2.face.LBPHFaceRecognizer_create()#$cv2.createLBPHFaceRecognizer()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    faces, Id = getImagesAndLabels("mydatasetconverted")
    recognizer.train(faces, np.array(Id))
    recognizer.save("recognizer2/trainningData2.yml")
    showinfo(title="Librarian Add", message="Congragulations!!! Data Trained")
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    images = []
    Ids = []
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        images.append(imageNp)
        Ids.append(Id)
    return images, Ids
trainImg = tk.Button(window, text="Train Images", command=TrainImages, fg="black", bg="grey", width=10, height=1,
                     activebackground="Red", font=('times', 15, ' bold '))
trainImg.place(x=200, y=500)

#----------------------------------------------------------------------------------

rec = cv2.face.LBPHFaceRecognizer_create()
rec.read("recognizer/trainningData.yml")
#path = 'dataSet'

id = 0
def getProfile(id):
    conn = sqlite3.connect("myDB.db")
    cmd="SELECT * FROM projects WHERE ID="+str(id)
    cursor = conn.execute(cmd)
    profile = None
    for row in cursor:
        profile = row
    conn.close()
    return profile
def browse():
    myfile = filedialog.askopenfilename()  # asking user to load an image
    cam = cv2.imread(myfile)
    resize = cv2.resize(cam,(520,520))
    gray = cv2.cvtColor(resize,cv2.COLOR_BGR2GRAY)
    faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = faceDetect.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        id, conf = rec.predict(gray[y:y + h, x:x + w])
        print(conf)
        def confidence(confi):
            confi = int(75*(confi/100))
            return confi
        confd = confidence (conf)
        cv2.rectangle(resize,(x,y),(x+w,y+h),(0,0,255),2)
        if confd >= 70:
            profile = getProfile(id)
        else:
            profile = 'No Profile'
        print(profile)
        print(confd)
    #    if profile != None:
        if confd >= 70:
            cv2.putText(resize,str(profile[1]),(x,y+h+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
    #        cv2.putText(resize,str(profile[2]), (x, y + h+60), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
    #       cv2.putText(resize,str(profile[3]), (x, y + h+90), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
    #        cv2.putText(resize,str(profile[4]), (x, y + h+120), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        else:
    #        print('not found')
            cv2.putText(resize,str('UnWanted'),(x,y+h+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
        cv2.imshow("Faces",resize)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
takeImg = tk.Button(window, text="Take Images", command=lambda: entry_Data(ID.get(),Name.get()), fg="black", bg="white", width=15, height=2,
                    activebackground="Red", font=('times', 15, ' bold '))
takeImg.place(x=0, y=300)
conversion = tk.Button(window, text="Conversion", command=sketch_Conversion, fg="black", bg="white", width=15, height=2,
                    activebackground="Red", font=('times', 15, ' bold '))
conversion.place(x=0, y=400)
loadImg = tk.Button(window, text="Open Image", command=browse, fg="black", bg="white", width=15, height=2,
                   activebackground="white", font=('times', 15, ' bold '))
loadImg.place(x=0, y=500)
quitWindow = tk.Button(window, text="Quit", command=window.destroy, fg="black", bg="white", width=15, height=2,
                       activebackground="white", font=('times', 15, ' bold '))
quitWindow.place(x=0, y=600)
label.pack()
id_label = Label(window,text='ID',font=20,bg='Yellow',fg='black')
ID = IntVar()
id_label_entry = Entry(window,textvariable=ID,bg='Yellow')
id_label_entry.place(x=0,y=150)
id_label.place(x=0, y=100)
name_label = Label(window,text='NAME',font=20,bg='Yellow',fg='black')
name_label.pack(pady=10,padx=10)
Name = StringVar()
name_label_entry = Entry(window,textvariable=Name)
name_label.place(x=0, y=200)
name_label_entry.place(x=0,y=250)
window.mainloop()

"""




























"""


import cv2
import os
import tkinter as tk
from tkinter import *
import sqlite3
from tkinter import filedialog

window = tk.Tk()
# helv36 = tk.Font(family='Helvetica', size=36, weight='bold')
window.title("Face_Recogniser")
window.geometry("1500x700")
photo = PhotoImage(file='BG1.png')
label = Label(window,image=photo)
#window.configure(background='Yellow')
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

cascade = ''
def loadCascade(cascading):
    cascading = 'haarcascade_frontalface_default.xml'
    classifer = cv2.CascadeClassifier(cascading)
    return classifer
camera = 1
def cameraCapture(cam):
    video = cv2.VideoCapture(cam)
    return video

def detection():
    cascadeCall = loadCascade(cascade)
    videoCall = cameraCapture(camera)

    while True:
        red,frame = videoCall.read()
        face = cascadeCall.detectMultiScale(frame,1.3,5)
        for x,y,w,h in face:
            def draw_rectangle(x_is,y_is,width,height):
                rectangle = cv2.rectangle(frame,(x_is,y_is),(x_is+width,y_is+height),(0,0,0),2)
                return rectangle
            draw_rectangle_Call = draw_rectangle(x,y,w,h)
            cv2.imshow("Video",frame)
        if cv2.waitKey(1)==ord('q'):
            break
    videoCall.release()
    cv2.destroyAllWindows()


ging = os.listdir("C:/Users/MUHAMMAD BILAL ZAFAR/PycharmProjects/ChikonEye_a-third-eye-app-which-protects-your-work-from-peepers.-ashraf-minhaj-patch-1/dataset")
print(ging)
sampleNum = 0

def GrayScale(rgb):
    r = rgb[:, :, 2]
    g = rgb[:, :, 2]
    b = rgb[:, :, 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray

def dodge(front, back):
    result = front * 255 / (255 - back)
    result[result > 255] = 255
    result[back == 255] = 255
    return result.astype('uint8')

def sketch_Conversion():
    sampleNum = 0
    for ginglement in ging:
        sampleNum = sampleNum + 1
        img = cv2.imread("C:/Users/MUHAMMAD BILAL ZAFAR/PycharmProjects/ChikonEye_a-third-eye-app-which-protects-your-work-from-peepers.-ashraf-minhaj-patch-1/dataset/" + ginglement)
        srcImgRZ = cv2.resize(img, (400, 400))
        img_gray = GrayScale(img)
        inverted_img = 255 - img_gray
        blur_img = cv2.GaussianBlur(inverted_img, ksize=(221, 221), sigmaX=75, sigmaY=75)
        final_img = dodge(blur_img, img_gray)
        cv2.imwrite("dataSetConvertIntoSketch2/" + ginglement, final_img)
        cv2.imshow('Image STEP IV', final_img)
        cv2.waitKey(100)
        cv2.destroyAllWindows()
    print('Successfully Conversion Done!!!!!')

rec = cv2.face.LBPHFaceRecognizer_create()

rec.read("recognizer/trainningData.yml")

#path = 'dataSet'

id = 0

def getProfile(id):
    conn = sqlite3.connect("FaceBase1.db")

    cmd="SELECT * FROM People1 WHERE ID="+str(id)

    cursor = conn.execute(cmd)

    profile = None

    for row in cursor:

        profile = row

    conn.close()

    return profile

def browse():

    myfile = filedialog.askopenfilename()  # asking user to load an image

    cam = cv2.imread(myfile)

    resize = cv2.resize(cam,(520,520))

    gray = cv2.cvtColor(resize,cv2.COLOR_BGR2GRAY)

    faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    faces = faceDetect.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:

        id, conf = rec.predict(gray[y:y + h, x:x + w])

        print(conf)

        def confidence(confi):

            confi = int(75*(confi/100))

            return confi

        confd = confidence (conf)

        cv2.rectangle(resize,(x,y),(x+w,y+h),(0,0,255),2)

        if confd >= 70:

            profile = getProfile(id)

        else:

            profile = 'No Profile'

        print(profile)

        print(confd)
    #    if profile != None:

        if confd >= 70:

            cv2.putText(resize,str(profile[1]),(x,y+h+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
    #        cv2.putText(resize,str(profile[2]), (x, y + h+60), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
    #       cv2.putText(resize,str(profile[3]), (x, y + h+90), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
    #        cv2.putText(resize,str(profile[4]), (x, y + h+120), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        else:
    #        print('not found')

            cv2.putText(resize,str('UnWanted'),(x,y+h+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)

        cv2.imshow("Faces",resize)

        cv2.waitKey(0)

        cv2.destroyAllWindows()

takeImg = tk.Button(window, text="Take Images", command=detection, fg="black", bg="white", width=15, height=2,
                    activebackground="Red", font=('times', 15, ' bold '))
takeImg.place(x=0, y=300)

conversion = tk.Button(window, text="Conversion", command=sketch_Conversion, fg="black", bg="white", width=15, height=2,
                    activebackground="Red", font=('times', 15, ' bold '))
conversion.place(x=0, y=400)

Search = tk.Button(window, text="Open Image", command=browse, fg="black", bg="white", width=15, height=2,
                   activebackground="white", font=('times', 15, ' bold '))

Search.place(x=0, y=500)

quitWindow = tk.Button(window, text="Quit", command=window.destroy, fg="black", bg="white", width=15, height=2,
                       activebackground="white", font=('times', 15, ' bold '))
quitWindow.place(x=0, y=600)

label.pack()
window.mainloop()



"""




