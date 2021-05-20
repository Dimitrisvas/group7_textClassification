### VERY IMPORTANT
from flask_implementation.model import fake_function, preprocess_data_without_pickle
### DO NOT REMOVE THIS IMPORT ^, it is used in prediction.py
import os
import logging
from flask import Flask, render_template

# Set logging level for debugging
logging.basicConfig(level=logging.INFO)

# Initialize app
app = None

# Create the app
def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    
    # Set environment
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flask.sqlite'),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
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

    # Initialize db with app
    from . import db
    db.init_app(app)

    # Register 'templates/comment' as a set of views
    from . import comment
    app.register_blueprint(comment.bp)

    # Register 'templates/prediction' as a set of views
    from . import prediction
    app.register_blueprint(prediction.bp)
    #app.add_url_rule('/', endpoint='index')

    return app

# If running app from __init__
if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)