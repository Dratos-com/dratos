from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    pass
import os
from dotenv import load_dotenv
from typing import Optional
import httpx

import daft
from daft.io import IOConfig, S3Config, GCSConfig, AzureConfig
import ray
from openai import AsyncOpenAI, OpenAI

# from unitycatalog import AsyncUnitycatalog, DefaultHttpxClient
import mlflow

# import tritonserver
from pyiceberg.catalog import load_catalog
from api.session import Session
from beta.services.tool_manager import tool_manager
from beta.models.base import BaseDBModel
from beta.tools.emotiv_tool import EmotivTool

# Firebase and Cloudflare imports are conditional
firebase_admin = None
CloudflareAPI = None


class Config:
    _instance = None

    def __init__(self, is_async: bool = True):
        load_dotenv()
        self.is_async = is_async
        self._load_environment_variables()
        self._initialize_services()

    def _load_environment_variables(self):
        # Load all environment variables
        self.UNITY_CATALOG_URL = os.environ.get("UNITY_CATALOG_URL")
        self.UNITY_CATALOG_TOKEN = os.environ.get("UNITY_CATALOG_TOKEN")
        self.RAY_RUNNER_HEAD = os.environ.get("RAY_RUNNER_HEAD")
        self.MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
        self.TRITON_MODEL_REPO = os.environ.get("TRITON_MODEL_REPO")
        self.FIREBASE_CREDENTIALS = os.environ.get("FIREBASE_CREDENTIALS")
        self.CLOUDFLARE_API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN")
        self.CLOUDFLARE_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
        self.ICEBERG_CATALOG_URI = os.environ.get("ICEBERG_CATALOG_URI")
        self.ICEBERG_WAREHOUSE = os.environ.get("ICEBERG_WAREHOUSE")

        # Emotiv-related environment variables
        self.EMOTIV_CLIENT_ID = os.environ.get("EMOTIV_CLIENT_ID")
        self.EMOTIV_CLIENT_SECRET = os.environ.get("EMOTIV_CLIENT_SECRET")
        self.EMOTIV_LICENSE = os.environ.get("EMOTIV_LICENSE")

        # Daft-related environment variables
        self.DAFT_RUNNER = os.environ.get("DAFT_RUNNER", "py")
        self.RAY_ADDRESS = os.environ.get("RAY_ADDRESS")

        # Cloud storage-related environment variables
        self.AWS_REGION = os.environ.get("AWS_REGION")
        self.AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
        self.AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
        self.GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
        self.GOOGLE_APPLICATION_CREDENTIALS = os.environ.get(
            "GOOGLE_APPLICATION_CREDENTIALS"
        )
        self.AZURE_STORAGE_ACCOUNT = os.environ.get("AZURE_STORAGE_ACCOUNT")
        self.AZURE_STORAGE_KEY = os.environ.get("AZURE_STORAGE_KEY")

    def _initialize_services(self):
        self._initialize_firebase()
        self._initialize_cloudflare()
        self._initialize_iceberg()
        self.register_tool(EmotivTool())
        self._initialize_daft()

    def _initialize_firebase(self):
        if self.FIREBASE_CREDENTIALS:
            global firebase_admin
            if firebase_admin is None:
                import firebase_admin
                from firebase_admin import credentials, auth
            try:
                cred = credentials.Certificate(self.FIREBASE_CREDENTIALS)
                firebase_admin.initialize_app(cred)
            except Exception as e:
                print(f"Failed to initialize Firebase: {e}")

    def _initialize_cloudflare(self):
        if self.CLOUDFLARE_API_TOKEN and self.CLOUDFLARE_ACCOUNT_ID:
            global CloudflareAPI
            if CloudflareAPI is None:
                from cloudflare import CloudflareAPI
            try:
                self.cloudflare_api = CloudflareAPI(token=self.CLOUDFLARE_API_TOKEN)
            except Exception as e:
                print(f"Failed to initialize Cloudflare: {e}")

    def _initialize_iceberg(self):
        self.iceberg_catalog = load_catalog(
            "iceberg",
            **{
                "uri": os.environ.get("ICEBERG_CATALOG_URI"),
                "warehouse": os.environ.get("ICEBERG_WAREHOUSE"),
            },
        )

    def _initialize_daft(self):
        if self.DAFT_RUNNER.lower() == "ray":
            if self.RAY_ADDRESS:
                daft.context.set_runner_ray(address=self.RAY_ADDRESS)
            else:
                print(
                    "Warning: RAY_ADDRESS not set. Falling back to local Python runner."
                )
                daft.context.set_runner_py()
        else:
            daft.context.set_runner_py()

        # Configure Daft I/O
        io_config_kwargs = {}

        if any([self.AWS_REGION, self.AWS_ACCESS_KEY_ID, self.AWS_SECRET_ACCESS_KEY]):
            io_config_kwargs["s3"] = S3Config(
                region_name=self.AWS_REGION,
                key_id=self.AWS_ACCESS_KEY_ID,
                access_key=self.AWS_SECRET_ACCESS_KEY,
            )

        if self.GCP_PROJECT_ID or self.GOOGLE_APPLICATION_CREDENTIALS:
            io_config_kwargs["gcs"] = GCSConfig(
                project_id=self.GCP_PROJECT_ID,
                credentials=self.GOOGLE_APPLICATION_CREDENTIALS,
            )

        if self.AZURE_STORAGE_ACCOUNT or self.AZURE_STORAGE_KEY:
            io_config_kwargs["azure"] = AzureConfig(
                storage_account=self.AZURE_STORAGE_ACCOUNT,
                access_key=self.AZURE_STORAGE_KEY,
            )

        if io_config_kwargs:
            daft_io_config = IOConfig(**io_config_kwargs)
            daft.set_planning_config(default_io_config=daft_io_config)
        else:
            print(
                "Warning: No cloud storage credentials found. Daft I/O config not set."
            )

    def create_session(self, user_id: str) -> Session:
        session = Session(user_id=user_id)
        self.save_to_iceberg(session)
        return session

    def update_session(self, session: Session):
        self.save_to_iceberg(session)

    def save_to_iceberg(self, model: BaseDBModel):
        table = self.iceberg_catalog.load_table(f"sessions.{model.__tablename__}")
        table.append(model.dict())

    def deploy_to_cloudflare(self, script_name, script_content):
        if not hasattr(self, "cloudflare_api"):
            print("Cloudflare is not initialized")
            return None
        try:
            return self.cloudflare_api.workers.create(
                account_id=self.CLOUDFLARE_ACCOUNT_ID,
                name=script_name,
                script=script_content,
            )
        except Exception as e:
            print(f"Failed to deploy to Cloudflare: {e}")
            return None

    def get_unity_catalog(self) -> AsyncUnitycatalog:
        return AsyncUnitycatalog(
            base_url=self.UNITY_CATALOG_URL,
            token=self.UNITY_CATALOG_TOKEN,
            timeout=httpx.Timeout(60.0, read=5.0, write=10.0, connect=2.0),
            http_client=DefaultHttpxClient(
                proxies=os.environ.get("HTTP_PROXY"),
                transport=httpx.HTTPTransport(local_address="0.0.0.0"),
            ),
        )

    def get_ray(self) -> ray:
        ray.init(self.RAY_RUNNER_HEAD, runtime_env={"pip": ["getdaft"]})
        return ray

    def get_daft(self) -> daft:
        return daft

    def get_mlflow(self) -> mlflow:
        mlflow.set_tracking_uri(self.MLFLOW_TRACKING_URI)
        return mlflow

    def get_openai_proxy(
        self,
        engine: str = "openai",
        is_async: Optional[bool] = None,
    ) -> AsyncOpenAI | OpenAI:
        if is_async is None:
            is_async = self.is_async
        return self.get_client(engine)

    def get_triton(self) -> tritonserver.Server:
        return tritonserver.Server(
            model_repository=self.TRITON_MODEL_REPO,
            model_control_mode=tritonserver.ModelControlMode.EXPLICIT,
            log_info=False,
        )

    def deploy_autosave_worker(self):
        script_content = """
        addEventListener('fetch', event => {
          event.respondWith(handleRequest(event.request))
        })

        async function handleRequest(request) {
          if (request.method === 'POST') {
            const data = await request.json()
            // Here you would typically encrypt the data with the user's key
            // For demonstration, we're just echoing it back
            return new Response(JSON.stringify(data), {
              headers: { 'Content-Type': 'application/json' }
            })
          }
          return new Response('Send a POST request with JSON data to autosave', { status: 200 })
        }
        """
        return self.deploy_to_cloudflare("autosave-worker", script_content)


config = Config.get_instance()


if __name__ == "__main__":
    # Usage example
    config = Config.get_instance()

    # Initialize Ray
    ray = config.get_ray()
    print("Ray initialized:", ray.is_initialized())

    # Get Daft
    daft = config.get_daft()
    print("Daft version:", daft.__version__)

    # Set up MLflow
    mlflow = config.get_mlflow()
    print("MLflow tracking URI:", mlflow.get_tracking_uri())

    # Get OpenAI proxy
    openai_async = config.get_openai_proxy(engine="openai", is_async=True)
    print("OpenAI async client:", type(openai_async))

    openai_sync = config.get_openai_proxy(engine="openai", is_async=False)
    print("OpenAI sync client:", type(openai_sync))

    # Initialize Triton server
    triton_server = config.get_triton()
    print("Triton server model repository:", triton_server.model_repository)

    # Deploy autosave worker (Note: This will actually deploy to Cloudflare if configured)
    # worker = config.deploy_autosave_worker()
    # print("Autosave worker deployed:", worker)

    # Clean up
    ray.shutdown()
    print("Ray shut down")
