MLflow-Ray-Serve deployment plugin
In this example, we will first train a model to classify the Iris dataset using sklearn. Next, we will deploy our model on Ray Serve and then scale it up, all using the MLflow Ray Serve plugin.

The plugin supports both a command line interface and a Python API. Below we will use the command line interface. For the full API documentation, see https://www.mlflow.org/docs/latest/cli.html#mlflow-deployments and https://www.mlflow.org/docs/latest/python_api/mlflow.deployments.html.

Plugin Installation
Please follow the installation instructions for the Ray Serve deployment plugin: https://github.com/ray-project/mlflow-ray-serve

Instructions
First, navigate to the directory for this example, mlflow/examples/ray_serve/.

Second, run python train_model.py. This trains and saves our classifier to the MLflow Model Registry and sets up automatic logging to MLflow. It also prints the mean squared error and the target names, which are species of iris:

MSE: 1.04
Target names:  ['setosa' 'versicolor' 'virginica']
Next, set the MLflow Tracking URI environment variable to the location where the Model Registry resides:

export MLFLOW_TRACKING_URI=sqlite:///mlruns.db

Now start a Ray cluster with the following command:

ray start --head

Next, start a long-running Ray Serve instance on your Ray cluster:

serve start

Ray Serve is now running and ready to deploy MLflow models. The MLflow Ray Serve plugin features both a Python API as well as a command-line interface. For this example, we'll use the command line interface.

Finally, we can deploy our model by creating an instance using the following command:

mlflow deployments create -t ray-serve -m models:/RayMLflowIntegration/1 --name iris:v1

The -t parameter here is the deployment target, which in our case is Ray Serve. The -m parameter is the Model URI, which consists of the registered model name and version in the Model Registry.

We can now run a prediction on our deployed model as follows. The file input.json contains a sample input containing the sepal length, sepal width, petal length, petal width of a sample flower. Now we can get the prediction using the following command:

mlflow deployments predict -t ray-serve --name iris:v1 --input-path input.json

This will output [0], [1], or [2], corresponding to the species listed above in the target names.

We can scale our deployed model up to use several replicas, improving throughput:

mlflow deployments update -t ray-serve --name iris:v1 --config num_replicas=2

Here we only used 2 replicas, but you can use as many as you like, depending on how many CPU cores are available in your Ray cluster.

The deployed model instance can be deleted as follows:

mlflow deployments delete -t ray-serve --name iris:v1

To tear down the Ray cluster, run the following command:

ray stop








MLflow-Ray-Serve
An experimental plugin that integrates Ray Serve with the MLflow pipeline. mlflow-ray-serve enables MLflow users to deploy MLflow models at scale on Ray Serve.

This plugin implements the Python API and command-line interface for MLflow deployment plugins.

Installation
pip install mlflow-ray-serve
The following packages are required and will be installed along with the plugin:

"ray[serve]"
"mlflow>=1.12.0"
This plugin requires Ray version 1.7.0 or greater.

Usage
This plugin must be used with a detached Ray Serve instance running on a Ray cluster. An easy way to set this up is by running the following two commands:

ray start --head # Start a single-node Ray cluster locally.
serve start # Start a detached Ray Serve instance.
The API is summarized below. For full details see the MLflow deployment plugin Python API and command-line interface documentation.

See https://github.com/mlflow/mlflow/tree/master/examples/ray_serve for a full example.

Create deployment
Deploy a model built with MLflow using Ray Serve with the desired configuration parameters; for example, num_replicas. Currently this plugin only supports the python_function flavor of MLflow models, and this is the default flavor.

CLI
mlflow deployments create -t ray-serve -m <model uri> --name <deployment name> -C num_replicas=<number of replicas>
Python API
from mlflow.deployments import get_deploy_client
target_uri = 'ray-serve'
plugin = get_deploy_client(target_uri)
plugin.create_deployment(
    name=<deployment name>,
    model_uri=<model uri>,
    config={"num_replicas": 4})
Update deployment
Modify the configuration of a deployed model and/or replace the deployment with a new model URI.

CLI
mlflow deployments update -t ray-serve --name <deployment name> -C num_replicas=<new number of replicas>
Python API
plugin.update_deployment(name=<deployment name>, config={"num_replicas": <new number of replicas>})
Delete deployment
Delete an existing deployment.

CLI
mlflow deployments delete -t ray-serve --name <deployment name>
Python API
plugin.delete_deployment(name=<deployment name>)
List deployments
List the names of all the models deployed on Ray Serve. Includes models not deployed via this plugin.

CLI
mlflow deployments list -t ray-serve
Python API
plugin.list_deployments()
Get deployment details
CLI
mlflow deployments get -t ray-serve --name <deployment name>
Python API
plugin.get_deployment(name=<deployment name>)
Run prediction on deployed model
For the prediction inputs, DataFrame, Tensor and JSON formats are supported by the Python API. To invoke via the command line, pass in the path to a JSON file containing the input.

CLI
mlflow deployments predict -t ray-serve --name <deployment name> --input-path <input file path> --output-path <output file path>
output-path is an optional parameter. Without it, the result will be printed in the terminal.

Python API
plugin.predict(name=<deployment name>, df=<prediction input>)
Plugin help
Prints the plugin help string.

CLI
mlflow deployments help -t ray-serve
