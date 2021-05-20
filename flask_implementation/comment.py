import logging
import sqlite3

from flask import Blueprint, render_template, request, url_for
from werkzeug.exceptions import abort

from .db import get_db

logging.basicConfig(level=logging.INFO)

bp = Blueprint('comment', __name__)

#global all_comments
all_comments = []

@bp.route('/comments', methods=['GET'])
def comments():

    #con = get_db()
    #db = con.cursor()
    
    db = get_db()
    global all_comments

    try:
        all_comments = db.execute(
            'SELECT comment_text, created'
            ' FROM comment'
            ' ORDER BY created DESC'
        ).fetchall()

    except sqlite3.Error as e:
        logging.error(e)
    except Exception as e:
        logging.error(e)

    return render_template('comment/comments.html', comments=all_comments)

#if __name__ == '__main__':
#    app.run(debug = True)