import sqlite3
import click
import logging

from flask import current_app, g
from flask.cli import with_appcontext

logging.basicConfig(level=logging.INFO)

def close_db(e=None):
    db = g.pop('db', None)

    if db is not None:
        db.close()

def get_db():
    if 'db' not in g:

        try:
            g.db = sqlite3.connect(
                current_app.config['DATABASE'],
                detect_types=sqlite3.PARSE_DECLTYPES
            )
            g.db.row_factory = sqlite3.Row
        
        except Exception as e:
            logging.error(e)

    return g.db

def init_app(app):
    app.teardown_appcontext(close_db)
    app.cli.add_command(init_db_command)

def init_db():
    db = get_db()

    with current_app.open_resource('flask_schema.sql') as f:
        db.executescript(f.read().decode('utf8'))


@click.command('init-db')
@with_appcontext
def init_db_command():
    """Clear the existing data and create new tables."""
    init_db()
    click.echo('Initialized the database.')