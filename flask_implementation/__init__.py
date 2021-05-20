from flask_implementation.model import fake_function, preprocess_data_without_pickle
import os
import logging
import sqlite3
from flask import Flask, render_template
from joblib import load
from os.path import join, dirname, realpath

logging.basicConfig(level=logging.INFO)

app = None
pipe = None
MODEL_PATH = join(dirname(realpath(__file__)), "multi_mnb_model.joblib")

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flask.sqlite'),
    )

    #if test_config is None:
        # load the instance config, if it exists, when not testing
    #    app.config.from_pyfile('config.py', silent=True)
    #else:
        # load the test config if passed in
    #    app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError as e:
        pass

    @app.route('/')
    def index():
        return render_template('index.html')
    
    #@app.route('/predictions', methods=['GET','POST'])
    #def predictions():
    #    return render_template("predictions.html")

    from . import db
    db.init_app(app)

    from . import comment
    app.register_blueprint(comment.bp)

    from . import prediction
    app.register_blueprint(prediction.bp)
    #app.add_url_rule('/', endpoint='index')

    #from . import model

    #global pipe
    #try:
    #    pipe = load(MODEL_PATH)
    #except ImportError as e:
    #    logging.error(e)

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)