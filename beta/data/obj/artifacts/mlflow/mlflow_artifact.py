from typing import List, Optional, Dict, Any
from beta.data.obj.artifacts import Artifact
import mlflow
from mlflow.tracking import MlflowClient
from api.config.context.clients.mlflow import MlflowConfig


class MlflowArtifactBridge:
    def __init__(self, mlflow_config: MlflowConfig):
        self.mlflow_config = mlflow_config
        self.client = MlflowClient(**mlflow_config.get_mlflow_client_kwargs())

    def log_artifact(
        self,
        artifact: Artifact,
        run_id: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ):
        """
        Log an artifact to MLflow.
        """
        if run_id is None:
            if experiment_name is None:
                experiment_name = f"{self.mlflow_config.experiment_name_prefix}default"
            experiment = self.get_or_create_experiment(experiment_name)
            run = self.client.create_run(experiment.experiment_id)
            run_id = run.info.run_id

        with mlflow.start_run(run_id=run_id):
            mlflow.log_dict(artifact.dict(), f"{artifact.id}_metadata.json")
            if artifact.payload:
                mlflow.log_artifact(
                    artifact.payload, f"{artifact.id}.{artifact.extension}"
                )

            # Log additional metadata
            mlflow.set_tag("artifact_id", artifact.id)
            mlflow.set_tag("artifact_name", artifact.name)
            mlflow.set_tag("artifact_type", artifact.mime_type)
            mlflow.set_tag("is_ai_generated", str(artifact.is_ai_generated))

        return run_id

    def get_or_create_experiment(
        self, experiment_name: str
    ) -> mlflow.entities.Experiment:
        """
        Get an existing experiment or create a new one if it doesn't exist.
        """
        experiment = self.client.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = self.client.create_experiment(
                experiment_name,
                artifact_location=self.mlflow_config.default_artifact_root,
            )
            experiment = self.client.get_experiment(experiment_id)
        return experiment

    def get_artifact(self, run_id: str, artifact_id: str) -> Optional[Artifact]:
        """
        Retrieve an artifact from MLflow.
        """
        run = self.client.get_run(run_id)
        artifact_uri = self.mlflow_config.get_artifact_uri(
            run_id, f"{artifact_id}_metadata.json"
        )

        try:
            artifact_dict = mlflow.artifacts.load_dict(artifact_uri)
            artifact = Artifact(**artifact_dict)

            # Load payload if it exists
            payload_uri = self.mlflow_config.get_artifact_uri(
                run_id, f"{artifact_id}.{artifact.extension}"
            )
            artifact.payload = mlflow.artifacts.load_artifact(payload_uri)

            return artifact
        except Exception as e:
            print(f"Error retrieving artifact: {e}")
            return None

    def list_artifacts(self, run_id: str) -> List[Artifact]:
        """
        List all artifacts for a given run.
        """
        artifacts = []
        for file_info in self.client.list_artifacts(run_id):
            if file_info.path.endswith("_metadata.json"):
                artifact_id = file_info.path.replace("_metadata.json", "")
                artifact = self.get_artifact(run_id, artifact_id)
                if artifact:
                    artifacts.append(artifact)
        return artifacts

    def delete_artifact(self, run_id: str, artifact_id: str):
        """
        Delete an artifact from MLflow.
        """
        self.client.delete_artifact(run_id, f"{artifact_id}_metadata.json")
        self.client.delete_artifact(run_id, f"{artifact_id}.{artifact.extension}")
