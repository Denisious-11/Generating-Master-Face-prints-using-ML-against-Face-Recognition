from tkinter import *
from PIL import ImageTk, Image
from tkinter.filedialog import askopenfilename
import cv2
import numpy as np
import argparse
import math
import glob
import tensorflow as tf
import os
from tkinter import messagebox
import warnings
warnings.filterwarnings('ignore')

from pred import perform_recognition

a = Tk()
a.title("Face Checker")
a.geometry("740x600")
a.minsize(740,600)
a.maxsize(740,600)


def get_output():


    list_box.insert(1, "Loading Image")
    list_box.insert(2, "")
    list_box.insert(3, "Preprocessing")
    list_box.insert(4, "")
    list_box.insert(5, "Face Detection")
    list_box.insert(6, "")
    list_box.insert(7, "Load Face Recognition System")
    list_box.insert(8, "")
    list_box.insert(9, "Prediction")

    get_result=perform_recognition(path)

    if get_result!="unknown":
        print("\nAttack Successful...")
        print("Recognized Person : ",get_result)
        messagebox.showinfo("Result","Attack Successful\n"+"Recognized Person :"+get_result)

    else:
        messagebox.showinfo("Result : ",get_result)


def Check():
    global f
    f.pack_forget()

    f = Frame(a, bg="white")
    f.pack(side="top", fill="both", expand=True)

    global f1
    f1 = Frame(f, bg="#97BC62")
    f1.place(x=0, y=0, width=500, height=690)
    f1.config()

    input_label = Label(f1, text="Input Image", font="arial 16", bg="#97BC62")
    input_label.place(x=200, y=20)    # input_label.pack(anchor=CENTER)

    upload_pic_button = Button(
        f1, text="Upload MasterFace", command=Upload, bg="#EA5D81")
    upload_pic_button.place(x=200, y=100)

    global label
    label=Label(f1,bg="#97BC62")

    f3 = Frame(f, bg="#2C5F2D")
    f3.place(x=500, y=0, width=240, height=690)
    f3.config()

    name_label = Label(f3, text="Process", font="arial 14", bg="#2C5F2D")
    name_label.pack(pady=20)

    global list_box
    list_box = Listbox(f3, height=12, width=31)
    list_box.pack()

    enhance_button = Button(
        f3, text="  Identify", command=get_output, bg="#F1E577")
    enhance_button.place(x=90, y=300)



def Upload():

    global path
    label.config(image='')
    list_box.delete(0,END)
    path = askopenfilename(title='Open a file',
                           initialdir='Test',
                           filetypes=(("JPG", "*.jpg"), ("JPEG", "*.jpeg"),("PNG", "*.png")))
    image = Image.open(path)
    global imagename
    imagename = ImageTk.PhotoImage(image.resize((300,300)))
    # label = Label(f1, image=imagename)
    label.config(image=imagename)
    label.image = imagename
    label.pack(anchor=CENTER,pady=180)


def Home():
    global f
    f.pack_forget()

    f = Frame(a, bg="Aquamarine")
    f.pack(side="top", fill="both", expand=True)

    front_image = Image.open("pic.jpg")
    front_photo = ImageTk.PhotoImage(front_image.resize((a.winfo_width(), a.winfo_height()), Image.ANTIALIAS))
    front_label = Label(f, image=front_photo)
    front_label.image = front_photo
    front_label.pack()

    home_label = Label(f, text="Face Checker",
                       font="arial 30",fg="red")
    home_label.place(x=255, y=160)


f = Frame(a, bg="Aquamarine")
f.pack(side="top", fill="both", expand=True)
front_image1 = Image.open("pic.jpg")
front_photo1 = ImageTk.PhotoImage(front_image1.resize((740, 600), Image.ANTIALIAS))
front_label1 = Label(f, image=front_photo1)
front_label1.image = front_photo1
front_label1.pack()

home_label = Label(f, text="Face Checker",
                   font="arial 30",fg="red")
home_label.place(x=255, y=160)

m = Menu(a)
m.add_command(label="Homepage", command=Home)
checkmenu = Menu(m)
m.add_command(label="Test Purpose", command=Check)

a.config(menu=m)


a.mainloop()
