MLflow Authentication Python API
mlflow.server.auth.client
classmlflow.server.auth.client.AuthServiceClient[source]
Bases: object

Client of an MLflow Tracking Server that enabled the default basic authentication plugin. It is recommended to use mlflow.server.get_app_client() to instantiate this class. See https://mlflow.org/docs/latest/auth.html for more information.

create_experiment_permission(experiment_id: str, username: str, permission: str)[source]
Create a permission on an experiment for a user.

Parameters
experiment_id – The id of the experiment.

username – The username.

permission – Permission to grant. Must be one of “READ”, “EDIT”, “MANAGE” and “NO_PERMISSIONS”.

Raises
mlflow.exceptions.RestException – if the user does not exist, or a permission already exists for this experiment user pair, or if the permission is invalid. Does not require experiment_id to be an existing experiment.

Returns
A single mlflow.server.auth.entities.ExperimentPermission object.

Example
export MLFLOW_TRACKING_USERNAME=admin
export MLFLOW_TRACKING_PASSWORD=password

from mlflow.server.auth.client import AuthServiceClient

client = AuthServiceClient("tracking_uri")
client.create_user("newuser", "newpassword")
ep = client.create_experiment_permission("myexperiment", "newuser", "READ")

print(f"experiment_id: {ep.experiment_id}")
print(f"user_id: {ep.user_id}")
print(f"permission: {ep.permission}")

Output
experiment_id: myexperiment
user_id: 3
permission: READ

create_registered_model_permission(name: str, username: str, permission: str)[source]
Create a permission on an registered model for a user.

Parameters
name – The name of the registered model.

username – The username.

permission – Permission to grant. Must be one of “READ”, “EDIT”, “MANAGE” and “NO_PERMISSIONS”.

Raises
mlflow.exceptions.RestException – if the user does not exist, or a permission already exists for this registered model user pair, or if the permission is invalid. Does not require name to be an existing registered model.

Returns
A single mlflow.server.auth.entities.RegisteredModelPermission object.

create_user(username: str, password: str)[source]
Create a new user.

Parameters
username – The username.

password – The user’s password. Must not be empty string.

Raises
mlflow.exceptions.RestException – if the username is already taken.

Returns
A single mlflow.server.auth.entities.User object.

Example
from mlflow.server.auth.client import AuthServiceClient

client = AuthServiceClient("tracking_uri")
user = client.create_user("newuser", "newpassword")
print(f"user_id: {user.id}")
print(f"username: {user.username}")
print(f"password_hash: {user.password_hash}")
print(f"is_admin: {user.is_admin}")

Output
user_id: 3
username: newuser
password_hash: REDACTED
is_admin: False

delete_experiment_permission(experiment_id: str, username: str)[source]
Delete an existing experiment permission for a user.

Parameters
experiment_id – The id of the experiment.

username – The username.

Raises
mlflow.exceptions.RestException – if the user does not exist, or no permission exists for this experiment user pair, or if the permission is invalid. Note that the default permission will still be effective even after the permission has been deleted.

Example
export MLFLOW_TRACKING_USERNAME=admin
export MLFLOW_TRACKING_PASSWORD=password

from mlflow.server.auth.client import AuthServiceClient

client = AuthServiceClient("tracking_uri")
client.create_user("newuser", "newpassword")
client.create_experiment_permission("myexperiment", "newuser", "READ")
client.delete_experiment_permission("myexperiment", "newuser")

delete_registered_model_permission(name: str, username: str)[source]
Delete an existing registered model permission for a user.

Parameters
name – The name of the registered model.

username – The username.

Raises
mlflow.exceptions.RestException – if the user does not exist, or no permission exists for this registered model user pair, or if the permission is invalid. Note that the default permission will still be effective even after the permission has been deleted.

Example
export MLFLOW_TRACKING_USERNAME=admin
export MLFLOW_TRACKING_PASSWORD=password

from mlflow.server.auth.client import AuthServiceClient

client = AuthServiceClient("tracking_uri")
client.create_user("newuser", "newpassword")
client.create_registered_model_permission("myregisteredmodel", "newuser", "READ")
client.delete_registered_model_permission("myregisteredmodel", "newuser")

delete_user(username: str)[source]
Delete a specific user.

Parameters
username – The username.

Raises
mlflow.exceptions.RestException – if the user does not exist

Example
export MLFLOW_TRACKING_USERNAME=admin
export MLFLOW_TRACKING_PASSWORD=password

from mlflow.server.auth.client import AuthServiceClient

client = AuthServiceClient("tracking_uri")
client.create_user("newuser", "newpassword")

client.delete_user("newuser")

get_experiment_permission(experiment_id: str, username: str)[source]
Get an experiment permission for a user.

Parameters
experiment_id – The id of the experiment.

username – The username.

Raises
mlflow.exceptions.RestException – if the user does not exist, or no permission exists for this experiment user pair. Note that the default permission will still be effective even if no permission exists.

Returns
A single mlflow.server.auth.entities.ExperimentPermission object.

Example
export MLFLOW_TRACKING_USERNAME=admin
export MLFLOW_TRACKING_PASSWORD=password

from mlflow.server.auth.client import AuthServiceClient

client = AuthServiceClient("tracking_uri")
client.create_user("newuser", "newpassword")
client.create_experiment_permission("myexperiment", "newuser", "READ")
ep = client.get_experiment_permission("myexperiment", "newuser")
print(f"experiment_id: {ep.experiment_id}")
print(f"user_id: {ep.user_id}")
print(f"permission: {ep.permission}")

Output
experiment_id: myexperiment
user_id: 3
permission: READ

get_registered_model_permission(name: str, username: str)[source]
Get an registered model permission for a user.

Parameters
name – The name of the registered model.

username – The username.

Raises
mlflow.exceptions.RestException – if the user does not exist, or no permission exists for this registered model user pair. Note that the default permission will still be effective even if no permission exists.

Returns
A single mlflow.server.auth.entities.RegisteredModelPermission object.

Example
export MLFLOW_TRACKING_USERNAME=admin
export MLFLOW_TRACKING_PASSWORD=password

from mlflow.server.auth.client import AuthServiceClient

client = AuthServiceClient("tracking_uri")
client.create_user("newuser", "newpassword")
client.create_registered_model_permission("myregisteredmodel", "newuser", "READ")
rmp = client.get_registered_model_permission("myregisteredmodel", "newuser")

print(f"name: {rmp.name}")
print(f"user_id: {rmp.user_id}")
print(f"permission: {rmp.permission}")

Output
name: myregisteredmodel
user_id: 3
permission: READ

get_user(username: str)[source]
Get a user with a specific username.

Parameters
username – The username.

Raises
mlflow.exceptions.RestException – if the user does not exist

Returns
A single mlflow.server.auth.entities.User object.

Example
export MLFLOW_TRACKING_USERNAME=admin
export MLFLOW_TRACKING_PASSWORD=password

from mlflow.server.auth.client import AuthServiceClient

client = AuthServiceClient("tracking_uri")
client.create_user("newuser", "newpassword")
user = client.get_user("newuser")

print(f"user_id: {user.id}")
print(f"username: {user.username}")
print(f"password_hash: {user.password_hash}")
print(f"is_admin: {user.is_admin}")

Output
user_id: 3
username: newuser
password_hash: REDACTED
is_admin: False

update_experiment_permission(experiment_id: str, username: str, permission: str)[source]
Update an existing experiment permission for a user.

Parameters
experiment_id – The id of the experiment.

username – The username.

permission – New permission to grant. Must be one of “READ”, “EDIT”, “MANAGE” and “NO_PERMISSIONS”.

Raises
mlflow.exceptions.RestException – if the user does not exist, or no permission exists for this experiment user pair, or if the permission is invalid

Example
export MLFLOW_TRACKING_USERNAME=admin
export MLFLOW_TRACKING_PASSWORD=password

from mlflow.server.auth.client import AuthServiceClient

client = AuthServiceClient("tracking_uri")
client.create_user("newuser", "newpassword")
client.create_experiment_permission("myexperiment", "newuser", "READ")
client.update_experiment_permission("myexperiment", "newuser", "EDIT")

update_registered_model_permission(name: str, username: str, permission: str)[source]
Update an existing registered model permission for a user.

Parameters
name – The name of the registered model.

username – The username.

permission – New permission to grant. Must be one of “READ”, “EDIT”, “MANAGE” and “NO_PERMISSIONS”.

Raises
mlflow.exceptions.RestException – if the user does not exist, or no permission exists for this registered model user pair, or if the permission is invalid.

Example
export MLFLOW_TRACKING_USERNAME=admin
export MLFLOW_TRACKING_PASSWORD=password

from mlflow.server.auth.client import AuthServiceClient

client = AuthServiceClient("tracking_uri")
client.create_user("newuser", "newpassword")
client.create_registered_model_permission("myregisteredmodel", "newuser", "READ")
client.update_registered_model_permission("myregisteredmodel", "newuser", "EDIT")

update_user_admin(username: str, is_admin: bool)[source]
Update the admin status of a specific user.

Parameters
username – The username.

is_admin – The new admin status.

Raises
mlflow.exceptions.RestException – if the user does not exist

Example
export MLFLOW_TRACKING_USERNAME=admin
export MLFLOW_TRACKING_PASSWORD=password

from mlflow.server.auth.client import AuthServiceClient

client = AuthServiceClient("tracking_uri")
client.create_user("newuser", "newpassword")

client.update_user_admin("newuser", True)

update_user_password(username: str, password: str)[source]
Update the password of a specific user.

Parameters
username – The username.

password – The new password.

Raises
mlflow.exceptions.RestException – if the user does not exist

Example
export MLFLOW_TRACKING_USERNAME=admin
export MLFLOW_TRACKING_PASSWORD=password

from mlflow.server.auth.client import AuthServiceClient

client = AuthServiceClient("tracking_uri")
client.create_user("newuser", "newpassword")

client.update_user_password("newuser", "anotherpassword")

mlflow.server.auth.entities
classmlflow.server.auth.entities.ExperimentPermission(experiment_id, user_id, permission)[source]
Bases: object

propertyexperiment_id
classmethodfrom_json(dictionary)[source]
propertypermission
to_json()[source]
propertyuser_id
classmlflow.server.auth.entities.RegisteredModelPermission(name, user_id, permission)[source]
Bases: object

classmethodfrom_json(dictionary)[source]
propertyname
propertypermission
to_json()[source]
propertyuser_id
classmlflow.server.auth.entities.User(id_, username, password_hash, is_admin, experiment_permissions=None, registered_model_permissions=None)[source]
Bases: object

propertyexperiment_permissions
classmethodfrom_json(dictionary)[source]
propertyid
propertyis_admin
propertypassword_hash
propertyregistered_model_permissions
to_json()[source]
propertyusername



MLflow Authentication REST API
The MLflow Authentication REST API allows you to create, get, update and delete users, experiment permissions and registered model permissions. The API is hosted under the /api route on the MLflow tracking server. For example, to list experiments on a tracking server hosted at http://localhost:5000, access http://localhost:5000/api/2.0/mlflow/users/create.

Important

The MLflow REST API requires content type application/json for all POST requests.

Table of Contents

Create User

Get User

Update User Password

Update User Admin

Delete User

Create Experiment Permission

Get Experiment Permission

Update Experiment Permission

Delete Experiment Permission

Create Registered Model Permission

Get Registered Model Permission

Update Registered Model Permission

Delete Registered Model Permission

Data Structures

Create User
Endpoint

HTTP Method

2.0/mlflow/users/create

POST

Request Structure
Field Name

Type

Description

username

STRING

Username.

password

STRING

Password.

Response Structure
Field Name

Type

Description

user

User

A user object.

Get User
Endpoint

HTTP Method

2.0/mlflow/users/get

GET

Request Structure
Field Name

Type

Description

username

STRING

Username.

Response Structure
Field Name

Type

Description

user

User

A user object.

Update User Password
Endpoint

HTTP Method

2.0/mlflow/users/update-password

PATCH

Request Structure
Field Name

Type

Description

username

STRING

Username.

password

STRING

New password.

Update User Admin
Endpoint

HTTP Method

2.0/mlflow/users/update-admin

PATCH

Request Structure
Field Name

Type

Description

username

STRING

Username.

is_admin

BOOLEAN

New admin status.

Delete User
Endpoint

HTTP Method

2.0/mlflow/users/delete

DELETE

Request Structure
Field Name

Type

Description

username

STRING

Username.

Create Experiment Permission
Endpoint

HTTP Method

2.0/mlflow/experiments/permissions/create

POST

Request Structure
Field Name

Type

Description

experiment_id

STRING

Experiment id.

username

STRING

Username.

permission

Permission

Permission to grant.

Response Structure
Field Name

Type

Description

experiment_permission

ExperimentPermission

An experiment permission object.

Get Experiment Permission
Endpoint

HTTP Method

2.0/mlflow/experiments/permissions/get

GET

Request Structure
Field Name

Type

Description

experiment_id

STRING

Experiment id.

username

STRING

Username.

Response Structure
Field Name

Type

Description

experiment_permission

ExperimentPermission

An experiment permission object.

Update Experiment Permission
Endpoint

HTTP Method

2.0/mlflow/experiments/permissions/update

PATCH

Request Structure
Field Name

Type

Description

experiment_id

STRING

Experiment id.

username

STRING

Username.

permission

Permission

New permission to grant.

Delete Experiment Permission
Endpoint

HTTP Method

2.0/mlflow/experiments/permissions/delete

DELETE

Request Structure
Field Name

Type

Description

experiment_id

STRING

Experiment id.

username

STRING

Username.

Create Registered Model Permission
Endpoint

HTTP Method

2.0/mlflow/registered-models/permissions/create

CREATE

Request Structure
Field Name

Type

Description

name

STRING

Registered model name.

username

STRING

Username.

permission

Permission

Permission to grant.

Response Structure
Field Name

Type

Description

registered_model_permission

RegisteredModelPermission

A registered model permission object.

Get Registered Model Permission
Endpoint

HTTP Method

2.0/mlflow/registered-models/permissions/get

GET

Request Structure
Field Name

Type

Description

name

STRING

Registered model name.

username

STRING

Username.

Response Structure
Field Name

Type

Description

registered_model_permission

RegisteredModelPermission

A registered model permission object.

Update Registered Model Permission
Endpoint

HTTP Method

2.0/mlflow/registered-models/permissions/update

PATCH

Request Structure
Field Name

Type

Description

name

STRING

Registered model name.

username

STRING

Username.

permission

Permission

New permission to grant.

Delete Registered Model Permission
Endpoint

HTTP Method

2.0/mlflow/registered-models/permissions/delete

DELETE

Request Structure
Field Name

Type

Description

name

STRING

Registered model name.

username

STRING

Username.

Data Structures
User
Field Name

Type

Description

id

STRING

User ID.

username

STRING

Username.

is_admin

BOOLEAN

Whether the user is an admin.

experiment_permissions

An array of ExperimentPermission

All experiment permissions explicitly granted to the user.

registered_model_permissions

An array of RegisteredModelPermission

All registered model permissions explicitly granted to the user.

Permission
Permission of a user to an experiment or a registered model.

Name

Description

READ

Can read.

EDIT

Can read and update.

MANAGE

Can read, update, delete and manage.

NO_PERMISSIONS

No permissions.

ExperimentPermission
Field Name

Type

Description

experiment_id

STRING

Experiment id.

user_id

STRING

User id.

permission

Permission

Permission granted.

RegisteredModelPermission
Field Name

Type

Description

name

STRING

Registered model name.

user_id

STRING

User id.

permission

Permission

Permission granted.

