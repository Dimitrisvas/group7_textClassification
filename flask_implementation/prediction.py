import logging
import pandas as pd
import sqlite3

from flask import Flask, render_template, request, redirect, url_for, Blueprint, flash
from joblib import load
from flask_implementation.db import get_db
from os.path import join, dirname, realpath

logging.basicConfig(level=logging.INFO)

MODEL_PATH = join(dirname(realpath(__file__)), "multi_mnb_model.joblib")
bp = Blueprint('prediction', __name__)

labels = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
cols = ['comment_text','toxic','severe_toxic','obscene','threat','insult','identity_hate']

df = pd.DataFrame(columns=cols)

#if __name__=='__main__':
try:
    pipe = load(MODEL_PATH)
except IOError as e:
    logging.error(e)
except ImportError as e:
    logging.error(e)
except Exception as e:
    logging.error(e)

# convert to df to feed into pipeline
def convert_for_pred(comment):

    temp_df = pd.DataFrame(columns=cols)

    new_row = {'comment_text':comment}

    for i in range(len(labels)):
        new_row[labels[i]] = 0

    temp_df = temp_df.append(new_row, ignore_index=True)

    return temp_df

@bp.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        comment = request.form['comment_text']
        error = None

        if not comment:
            error = 'Comment is required.'

        if error is not None:
            flash(error)
        else:

            toxic = False
            comment = convert_for_pred(comment)

            prediction = pipe.predict(comment['comment_text']).tolist()
            logging.info(prediction)

            new_row = {'comment_text':comment['comment_text']}

            for i in range(len(labels)):
                new_row[labels[i]] = prediction[0][i]

                if prediction[0][i] == 1:
                    toxic = True

            #append row to the dataframe
            global df
            df = df.append(new_row, ignore_index=True)
            
            if not toxic:
            
                try:
                    db = get_db()
                    db.execute(
                        'INSERT INTO comment (comment_text)'
                        ' VALUES (?)',
                        (comment)
                    )
                    db.commit()
                    return redirect('prediction/predictions.html')

                except sqlite3.Error as e:
                    logging.error(e)

    return render_template('index.html')

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


#if __name__ == '__main__' :
#    app.run(debug=True)