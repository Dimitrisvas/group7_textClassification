import sqlite3
import click
import logging

from flask import current_app, g
from flask.cli import with_appcontext

# Set logging level for debugging
logging.basicConfig(level=logging.INFO)

# According to flask docs, "g is a special object that is unique for each request."

# Close db
def close_db(e=None):
    db = g.pop('db', None)

    if db is not None:
        db.close()

# Establish db connection
def get_db():
    if 'db' not in g:

        try:
            g.db = sqlite3.connect("./instance/database.db")
            g.db.row_factory = sqlite3.Row
        
        except Exception as e:
            logging.error(e)

    return g.db

# Close db connection and adds command line function
def init_app(app):
    app.teardown_appcontext(close_db)
    app.cli.add_command(init_db_command)

# Create db and tables from flask_schema.sql
def init_db():
    db = get_db()

    with current_app.open_resource('flask_schema.sql') as f:
        db.executescript(f.read().decode('utf8'))

# Set command line command init-db
@click.command('init-db')
@with_appcontext
def init_db_command():
    """Clear the existing data and create new tables."""
    init_db()
    click.echo('Initialized the database.')