import os
import tempfile

import pytest

from flask_implementation import create_app


@pytest.fixture
def client():
    app = create_app()
    db_fd, app.config['DATABASE'] = tempfile.mkstemp()
    app.config['TESTING'] = True

    with app.test_client() as client:
    #    with app.app_context():
    #        init_db()
        yield client

    os.close(db_fd)
    os.unlink(app.config['DATABASE'])

def submit_comment(client, comment):
    return client.post('/', data=dict(
        comment_text=comment,
        formGroupExampleInput2=""
    ), follow_redirects=True)

def goto_predictions(client):
    return client.get('/predictions')

def test_prediction(client):
    comment = "you suck"

    rv = submit_comment(client, comment)

    rv = goto_predictions(client)
    assert b'you suck' in rv.data