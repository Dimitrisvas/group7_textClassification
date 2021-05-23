import logging
import pandas as pd
import sqlite3

from flask import Flask, render_template, request, redirect, url_for, Blueprint, flash
from joblib import load
from os.path import join, dirname, realpath

from flask_implementation.db import get_db

# Set logging level for debugging
logging.basicConfig(level=logging.INFO)

# Pipeline file path
MODEL_PATH = join(dirname(realpath(__file__)), "multi_mnb_model.joblib")

# Set blueprint
bp = Blueprint('prediction', __name__)

# List of labels the model is trained on
labels = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
# List of columns to add to dataframe
cols = ['comment_text','toxic','severe_toxic','obscene','threat','insult','identity_hate']
# Initialize dataframe
df = pd.DataFrame(columns=cols)

try:
    pipe = load(MODEL_PATH)
except IOError as e:
    logging.error(e)
except ImportError as e:
    logging.error(e)
except Exception as e:
    logging.error(e)

# Convert to dataframe to feed into pipeline
# Params:
#   String - @comment: Comment from form input
# Output:
#   Dataframe - @comment_df: Dataframe with 7 columns with comment
def convert_for_pred(comment):

    comment_df = pd.DataFrame(columns=cols)

    # New dict to add to dataframe
    new_row = {'comment_text':comment}

    # Initialize all label values as 0
    for i in range(len(labels)):
        new_row[labels[i]] = 0

    # Append row to dataframe
    comment_df = comment_df.append(new_row, ignore_index=True)

    return comment_df

# Set route and method, function name has to match route and .html file name
@bp.route('/', methods=['GET', 'POST'])
def index():

    if request.method == 'POST':
        # Get input from input name = 'comment_text'
        comment = request.form['comment_text']
        # Error check
        error = None

        # If comment is empty (currently uses html's 'required' thing so this probably isnt needed)
        if not comment:
            error = 'Comment is required.'

        # If error
        if error is not None:
            flash(error)
        
        # If no error
        else:
            # Initialize boolean for whether comment is toxic
            toxic = False
            # Dataframe of comment input
            comment_df = convert_for_pred(comment)

            # Predict comment
            prediction = pipe.predict(comment_df['comment_text']).tolist()

            # New dict to add to dataframe
            new_row = {'comment_text':comment}

            # Add actual prediction values
            for i in range(len(labels)):
                new_row[labels[i]] = prediction[0][i]
                # If value of any label is 1, comment is toxic
                if prediction[0][i] == 1:
                    toxic = True

            # Append row to dataframe
            global df # global to pass to /predictions.html
            df = df.append(new_row, ignore_index=True)
            
            try:
                con = sqlite3.connect("./instance/database.db")
                cur = con.cursor()
                cur.execute(
                    'INSERT INTO comment(comment_text,toxic,severe_toxic,obscene,threat,insult,identity_hate)'
                    ' VALUES (?,?,?,?,?,?,?)',
                    (comment,prediction[0][0],prediction[0][1],prediction[0][2],
                     prediction[0][3],prediction[0][4],prediction[0][5],)
                )
                con.commit()

                return redirect('predictions')

            except sqlite3.Error as e:
                logging.error(e)

    return render_template('index.html')

# Set route and method, function name has to match route and .html file name
# Renders dataframe in predictions.html
@bp.route('/predictions', methods=('GET','POST'))
def predictions():
    return render_template("prediction/predictions.html", data=df.to_html())

#@app.route('/', methods=['POST', 'GET'])
#def get_data():
#    if request.method == 'POST':
#        user = request.form['search']
#        return redirect(url_for('success', name=user))


#@app.route('/success/<name>')
#def success(name):
#    return "<xmp>" + str(requestResults(name)) + " </xmp> "