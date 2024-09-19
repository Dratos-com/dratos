Connecting to a Centralized Database
By default, MLflow Authentication uses a local SQLite database to store user and permission data. In the case of a multi-node deployment, it is recommended to use a centralized database to store this data.

To connect to a centralized database, you can set the database_uri configuration variable to the database URL.

Example: /path/to/my_auth_config.ini
[mlflow]
database_uri = postgresql://username:password@hostname:port/database

Then, start the MLflow server with the MLFLOW_AUTH_CONFIG_PATH environment variable set to the path of your configuration file.

MLFLOW_AUTH_CONFIG_PATH=/path/to/my_auth_config.ini mlflow server --app-name basic-auth

The database must be created before starting the MLflow server. The database schema will be created automatically when the server starts.

Custom Authentication
MLflow authentication is designed to be extensible. If your organization desires more advanced authentication logic (e.g., token-based authentication), it is possible to install a third party plugin or to create your own plugin.

Your plugin should be an installable Python package. It should include an app factory that extends the MLflow app and, optionally, implement a client to manage permissions. The app factory function name will be passed to the --app argument in Flask CLI. See https://flask.palletsprojects.com/en/latest/cli/#application-discovery for more information.

Example: my_auth/__init__.py
from flask import Flask
from mlflow.server import app


def create_app(app: Flask = app):
    app.add_url_rule(...)
    return app


class MyAuthClient:
    ...

Then, the plugin should be installed in your Python environment:

pip install my_auth

Then, register your plugin in mlflow/setup.py:

setup(
    ...,
    entry_points="""
        ...

        [mlflow.app]
        my-auth=my_auth:create_app

        [mlflow.app.client]
        my-auth=my_auth:MyAuthClient
    """,
)

Then, you can start the MLflow server:

mlflow server --app-name my-auth