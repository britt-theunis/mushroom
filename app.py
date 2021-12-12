# app.py

import os

from flask import Flask, request, make_response, jsonify, render_template, redirect, flash
from flask.helpers import url_for
from werkzeug.utils import secure_filename
from fastai.vision.all import *
from fastai.data.external import *


# codeblock below is needed for Windows path #############
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
##########################################################

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)

# loading the pickle files from our 3 models
learner1 = load_learner('mushroom_noMushroom.pkl')
learner2 = load_learner('toxic_safe.pkl')
learner3 = load_learner('species.pkl')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():

    # if the form is submitted
    if request.method == 'POST':

        # check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']

        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename): 
            filename = secure_filename(file.filename)

            file.save('./static/' + filename)

            img = PILImage.create(file)
            
            # prediction whether it is a mushroom or not
            pred1 = learner1.predict(img)

            # if it is a mushroom
            if pred1[0] == "mushroom":
                # invoke models
                # accuracy (in %) that the model knows it is indeed a mushroom
                perc1 = round(float(pred1[2][0])*100, 2)

                # prediction of whether the mushroom is edible or not
                pred2 = learner2.predict(img)
                # accuracy (in %) that the model knows that the mushroom is edible or not
                perc2 = round(float(max(pred2[2]))*100, 2)
                
                # prediction of the mushroom species
                pred3 = learner3.predict(img)
                # accuracy (in %) that the model knows what the mushroom species is
                perc3 = round(float(max(pred3[2]))*100, 2)

                # go to the resultGood page with the variables defined above
                return render_template("resultGood.html", 
                filename=filename, 
                perc1 = perc1,
                variable2 = pred2[0], perc2 = perc2,
                variable3 = pred3[0], perc3 = perc3)

            # if it is not a mushroom
            if pred1[0] == "noMushroom":
                 # go to the resultFalse page
                return render_template("resultFalse.html", filename = filename)

    return render_template('home.html')