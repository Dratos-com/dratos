from typing import List, Optional, Dict, Any
from beta.data.obj.artifacts.artifact import Artifact
import mlflow
import pyarrow as pa


class MlflowArtifactAccessor:
    """
    Accessor for Artifact objects stored in MLflow.
    Provides methods for CRUD operations and data manipulation specific to MLflow Artifacts.
    """

    def __init__(self, experiment_id: str, tracking_uri: str = None):
        self.experiment_id = experiment_id
        self.tracking_uri = tracking_uri
        if self.tracking_uri is not None:
            mlflow.set_tracking_uri(self.tracking_uri)

    def get_artifacts(
        self, filter_expr: Optional[Dict[str, Any]] = None
    ) -> List[Artifact]:
        """Retrieve artifacts from MLflow, optionally filtered by a dictionary."""
        artifacts = []
        for run in mlflow.search_runs(
            experiment_ids=[self.experiment_id], filter_string=filter_expr
        ):
            for artifact_path in mlflow.list_artifacts(run.info.run_id):
                if artifact_path.path.endswith(".arrow"):
                    artifact_uri = mlflow.get_artifact_uri(
                        run.info.run_id, artifact_path.path
                    )
                    arrow_table = pa.ipc.open_file(artifact_uri).read_all()
                    artifacts.extend(Artifact.from_arrow_table(arrow_table))
        return artifacts

    def get_artifact_by_id(self, artifact_id: str) -> Artifact:
        """Retrieve a single artifact by its ID."""
        artifacts = self.get_artifacts(filter_expr={"tags.artifact_id": artifact_id})
        if artifacts:
            return artifacts[0]
        else:
            raise ValueError(f"Artifact with ID {artifact_id} not found.")

    def write_artifacts(self, artifacts: List[Artifact], run_name: str):
        """Write a list of artifacts to MLflow."""
        with mlflow.start_run(run_name=run_name, experiment_id=self.experiment_id):
            arrow_table = Artifact.to_arrow_table(artifacts)
            mlflow.log_artifact(
                arrow_table.to_feather("artifacts.arrow"), "artifacts.arrow"
            )
            mlflow.set_tag("artifact_id", artifacts[0].id)

    def update_artifact(self, artifact: Artifact, run_id: str):
        """Update a single artifact in MLflow."""
        with mlflow.start_run(run_id=run_id, experiment_id=self.experiment_id):
            arrow_table = Artifact.to_arrow_table([artifact])
            mlflow.log_artifact(
                arrow_table.to_feather("artifacts.arrow"), "artifacts.arrow"
            )
            mlflow.set_tag("artifact_id", artifact.id)

    def delete_artifact(self, artifact_id: str, run_id: str):
        """Delete a single artifact from MLflow."""
        mlflow.delete_artifact(run_id, "artifacts.arrow")

    def upsert_artifacts(self, artifacts: List[Artifact], run_name: str):
        """Upsert a list of artifacts to MLflow."""
        self.write_artifacts(artifacts, run_name)
