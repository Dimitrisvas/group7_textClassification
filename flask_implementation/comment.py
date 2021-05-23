import logging
import sqlite3
from flask import Blueprint, render_template, request, url_for
from werkzeug.exceptions import abort

from .db import get_db

# Set logging level for debugging
logging.basicConfig(level=logging.INFO)

# Set blueprint
bp = Blueprint('comment', __name__)

# Initialize list of comments to get from db
all_comments = []

# Set route and method, function name has to match route and .html file name
@bp.route('/comment', methods=['GET'])
def comments():

    # Connect to db
    #con = __init__.db
    con = sqlite3.connect("./instance/database.db") #get_db()
    db = con.cursor()

    global all_comments

    try:
        db.execute(
            'SELECT *'
            ' FROM comment'
            ' ORDER BY created DESC'
        )
        #db.execute("SELECT * FROM comment;")
        all_comments = db.fetchall()
        print(all_comments)

    except sqlite3.Error as e:
        logging.error(e)
    except Exception as e:
        logging.error(e)

    return render_template('comment/comments.html', comments=all_comments)