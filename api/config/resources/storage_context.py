from typing import Optional, Callable, Dict, Any
import os
from beta.data.obj.data_object import DataObject
from api.config.resources.storage_config import StorageConfig


class StorageContext(DataObject):
    def __init__(self, storage_config: StorageConfig):
        self._configure_environment_variables(storage_config)
        self._configure_clients()

    def _configure_environment_variables(self, sc: StorageConfig):
        """Set environment variables based on storage configurations."""
        
       if os.getenv('AWS_ACCESS_KEY_ID'): 
            s3_config = self.config_dict['storage']['s3']
            s3_config['region'] = os.getenv('AWS_REGION')
            s3_config['endpoint_url'] = os.getenv('AWS_ENDPOINT_URL')
            s3_config['access_key_id'] = os.getenv('AWS_ACCESS_KEY_ID')
            s3_config['secret_access_key'] = os.getenv('AWS_SECRET_ACCESS_KEY')
            s3_config['session_token'] = os.getenv('AWS_SESSION_TOKEN')
            s3_config['anonymous'] = os.getenv('AWS_ANONYMOUS')
            s3_config['virtual_hosted_style'] = os.getenv('AWS_VIRTUAL_HOSTED_STYLE')
            s3_config['s3_express'] = os.getenv('AWS_S3_EXPRESS')
            s3_config['server_side_encryption'] = os.getenv('AWS_SERVER_SIDE_ENCRYPTION')
            s3_config['sse_kms_key_id'] = os.getenv('AWS_SSE_KMS_KEY_ID')
            s3_config['sse_bucket_key_enabled'] = os.getenv('AWS_SSE_BUCKET_KEY_ENABLED')
            s3_config['expiry'] = os.getenv('AWS_EXPIRY')
        
            self.config_dict['storage']['s3'] = s3_config 

        # if GOOGLE_APPLICATION_CREDENTIALS is set, assume we are using GCS
        elif os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
            gcs_config = self.config_dict['storage']['gcs']
            gcs_config['project_id'] = os.getenv('GOOGLE_CLOUD_PROJECT')
            gcs_config['credentials'] = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            gcs_config['token'] = os.getenv('GOOGLE_OAUTH_ACCESS_TOKEN')
            gcs_config['anonymous'] = os.getenv('GOOGLE_ANONYMOUS')
            gcs_config['google_service_account'] = os.getenv('GOOGLE_SERVICE_ACCOUNT')
            gcs_config['google_service_account_key'] = os.getenv('GOOGLE_SERVICE_ACCOUNT_KEY')

            self.config_dict['storage']['gcs'] = gcs_config 

        # if AZURE_STORAGE_ACCESS_KEY is set, assume we are using Azure
        elif os.getenv('AZURE_STORAGE_ACCESS_KEY'):
            azure_config = self.config_dict['storage']['azure']
            azure_config['storage_account'] = os.getenv('AZURE_STORAGE_ACCOUNT')
            azure_config['access_key'] = os.getenv('AZURE_STORAGE_ACCESS_KEY')
            azure_config['sas_token'] = os.getenv('AZURE_STORAGE_SAS_TOKEN')
            azure_config['tenant_id'] = os.getenv('AZURE_TENANT_ID')
            azure_config['client_id'] = os.getenv('AZURE_CLIENT_ID')
            azure_config['client_secret'] = os.getenv('AZURE_CLIENT_SECRET')

            self.config_dict['storage']['azure'] = azure_config 

        else:
            print("No Object Storage configuration found, assuming local storage.")
            local_config = self.config_dict['storage']['local']
            local_config['local_path'] = os.getenv('LOCAL_STORAGE_PATH')
            local_config['anonymous'] = os.getenv('LOCAL_STORAGE_ANONYMOUS')
            self.config_dict['storage']['local'] = local_config


        if sc.lance:
            sc.lance.allow_http = os.environ.get(
                "LANCE_ALLOW_HTTP", sc.lance.allow_http
            )
            sc.lance.allow_invalid_certificates = os.environ.get(
                "LANCE_ALLOW_INVALID_CERTIFICATES", sc.lance.allow_invalid_certificates
            )
            sc.lance.connect_timeout = os.environ.get(
                "LANCE_CONNECT_TIMEOUT", sc.lance.connect_timeout
            )
            sc.lance.timeout = os.environ.get("LANCE_TIMEOUT", sc.lance.timeout)
            sc.lance.user_agent = os.environ.get(
                "LANCE_USER_AGENT", sc.lance.user_agent
            )
            sc.lance.proxy_url = os.environ.get("LANCE_PROXY_URL", sc.lance.proxy_url)
            sc.lance.proxy_ca_certificate = os.environ.get(
                "LANCE_PROXY_CA_CERTIFICATE", sc.lance.proxy_ca_certificate
            )
            sc.lance.proxy_excludes = os.environ.get(
                "LANCE_PROXY_EXCLUDES", sc.lance.proxy_excludes
            )

        return sc

    def _configure_clients(self, sc: StorageConfig):
        """Configure Daft, MLflow, and LanceDB clients."""
        self._configure_daft(sc)
        self._configure_mlflow()
        self._configure_lancedb()

    def _configure_daft(self, sc: StorageConfig):
        """Configure Daft client."""
        daft_context = DaftContext(sc)

    def _configure_mlflow(self):
        """Configure MLflow client."""
        pass

    def _configure_lancedb(self):
        """Configure LanceDB client."""
        pass
