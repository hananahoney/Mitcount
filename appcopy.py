from flask import Flask, render_template, request, flash,redirect,session,url_for, send_from_directory
import mysql.connector
import cv2 as cv
import keras_ocr
import re
import numpy as np
import tensorflow as tf
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
from PIL import Image
from transformers import *
from ultralytics import YOLO
import hashlib
import skimage
import skimage.segmentation
from skimage.segmentation import mark_boundaries
from skimage import data, segmentation
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def count_lines(file_path):
    try:
        files = [f for f in os.listdir(file_path) if f.endswith('.txt')]
        if len(files) == 0:
            print("No .txt files found in the directory.")
        else:
            for file_name in files:
                with open(os.path.join(file_path, file_name), 'r') as file:
                    lines = file.readlines()
                    num_lines = len(lines)
                    return num_lines
                    # print("Number of lines in", file_name, ":", num_lines)
    except FileNotFoundError:
        print("Directory not found.")
    except IOError:
        print("Error reading the file.")








app = Flask(__name__)
app.secret_key = "123"


model = YOLO('best.pt')

model2 = YOLO('yolov8n.pt')

# model = YOLO('npbest.pt')
# model.predict(
# source="myimg1.jpg",
#     conf=0.25,
#             save=True, save_txt=True
#         )



# MySQL configuration
eemail = 'hassanrana10@gmail.com'

mysql_host = 'localhost'
mysql_user = 'root'
mysql_password = ''
mysql_db = 'mbilal'
conn = mysql.connector.connect(host=mysql_host, user=mysql_user, password=mysql_password, database=mysql_db)
cursor = conn.cursor()
personid = 0
@app.route('/')
def index():
    return render_template('mylogin.html')



@app.route('/mylogin')
def signup():
    return render_template('myform.html')


@app.route('/patientrecords/<int:personid>')
def PatientRecord(personid):
    # cursor.execute("SELECT * FROM patientss WHERE patient_id = %s", (personid,))
    cursor.execute("SELECT * FROM patients WHERE patient_id = %s AND eemail = %s", (personid, eemail))

    data = cursor.fetchall()
    return render_template('patientrecords.html', data=data,personid=personid)

@app.route('/myform')
def signin():
    return render_template('mylogin.html')

@app.route('/details', methods=['POST'])
def details():
    data = request.get_json()
    personid = data['personid']

    print("Person id is = ",personid)
    return redirect(url_for('PatientRecord', personid=personid))

@app.route('/Records')
def Records():
    try:
        # cursor.execute("SELECT * FROM patient")
        # cursor.execute("select * FROM patientss GROUP by (patient_id)")
        cursor.execute("SELECT * FROM patients WHERE eemail = %s GROUP BY patient_id HAVING COUNT(*) >= 1", (eemail,))

        data = cursor.fetchall()
        return render_template('Records.html', data=data)
    except Exception as e:
        print("ERROR : " + str(e))

    flash("Error in Fetching Data")
    return redirect('Home')

@app.route('/Home')
def Home():
    return render_template('index.html', uploaded_image = None, resultant_image = None)

@app.route('/mylogin', methods=['POST','GET'])
def mylogin():
    if request.method == 'POST':

        global eemail

        eemail = request.form['email']
        password = request.form['password']
        query = "SELECT * FROM user WHERE email = %s"
        values = (eemail,)
        cursor.execute(query, values)
        user = cursor.fetchone()
        print("login password",password)
        if user:
            # Email exists in database, verify password
            hashed_password = user[3].encode('utf-8')
            print("Login hased password",hashed_password) 
            if user[3] == hashlib.sha256(password.encode('utf-8') + eemail.encode('utf-8')).hexdigest():
                print("Login hased password",hashed_password)
                flash('Valid email and password', 'success')
                return render_template('index.html')
            else:
                flash('Invalid Email and password', 'failure')
                return render_template('mylogin.html')
        else:
            flash('Invalid email or password', 'error')
            return render_template('mylogin.html')
    else:
        return render_template('mylogin.html')




@app.route('/register', methods=['POST', 'GET'])
def register():
    if request.method == 'POST':
        name = request.form['name']

        global eemail
        eemail = request.form['email']
        password = request.form['password']
        hashed_password = hashlib.sha256(password.encode('utf-8') + eemail.encode('utf-8')).hexdigest()

        query = "SELECT * FROM user WHERE email = %s"
        values = (eemail,)
        cursor.execute(query, values)
        result = cursor.fetchone()
        if result is not None:
            # Email already exists in database
            flash("Email already exists")
            return render_template('myform.html', message="Email already exists")
        else:
            # Email does not exist, insert new user into database
            query = "INSERT INTO user (name, email, password) VALUES (%s, %s, %s)"
            values = (name, eemail, hashed_password)
            cursor.execute(query, values)
            conn.commit()
            flash('Registration successful', 'success')
            return render_template('index.html')
    else:
        return render_template('myform.html')


@app.route('/login')
def login():
    message = request.args.get('message')
    return render_template('mylogin.html', message=message)


@app.route('/submit-form', methods=['POST'])
def handle_form_submission():
    if request.method == "POST":
        myImageFile = request.files['file']
        
        
        print(myImageFile)
        x=Image.open(myImageFile)
        x.save("2.jpg")
        
     
        
        patientImageName = myImageFile.filename
        
        machine = request.form.get('machine')


        if machine==('A'):
        
            results=model.predict(
            source="2.jpg",
                conf=0.25,
                        save=True, save_txt=True
                    )

            
            img_path = fr"D:\fypmodels\copy_mitwebb\runs\detect\predict\2.jpg"
            print("________________________")
            print(results)
            # Load the image using cv.imread()
            img = Image.open(img_path)
        
        
        elif machine==('H'):
            print("You are a senior citizen.") 
            
        
            image1 = "2.jpg"
            img = cv.imread(image1)
        
            img = cv.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
            ret, thresh1 = cv.threshold(img,50, 200, cv2.THRESH_BINARY)
            source = cv.GaussianBlur(thresh1, (7, 7), 0)

            mk=skimage.segmentation.mark_boundaries(img, source, color=(1, 1, 0), outline_color=None, mode='outer', background_label=0)

            mpimg.imsave('3.jpg' ,mk)







            results=model2.predict(
            source="3.jpg",
                conf=0.25,
                        save=True, save_txt=True
                    )

            
            img_path = fr"D:\fypmodels\copy_mitwebb\runs\detect\predict\3.jpg"
            print("________________________")
            print(results)
            # Load the image using cv.imread()
            img = Image.open(img_path)

            # myimage1 = "3.jpg"

            # # Read the image using Pillow
            # img = Image.open(myimage1)

            # # Display the image
            # img.show()

        else:
            print("no machine found.")    



















        name = request.form.get('name')
        patient_id = request.form.get('id')
        # doctorid=request.form.get('doctorid')
        prob="7"
        diagnostics="1"
        patientImageName = myImageFile.filename
        patientImageResultsName = "resultant " + patientImageName
        # machine = request.form.get('machine')

        x.save("static/upload_images/" + patientImageName)
        img.save("static/results_images/" + patientImageResultsName)
        
        
        
        try:        
            query = "INSERT INTO patients (patient_id, name, uploaded_image, results_image,prob,machine,diagnostics,time1,eemail) VALUES (%s, %s, %s,%s ,%s, %s,%s,NOW(),%s)"
            
            values = (patient_id, name, patientImageName, patientImageResultsName,prob,machine,diagnostics,eemail)
            cursor.execute(query, values)
            conn.commit()
        except Exception as e:
            print("Error Found : ", str(e))

    return render_template('index.html', uploaded_image = "/static/upload_images/" + patientImageName, resultant_image = "/static/results_images/" + patientImageResultsName)

if __name__ == '__main__':
    app.run('0.0.0.0',port=5000,debug=True)
    
    