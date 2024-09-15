from typing import Optional
from beta.data.obj import DataObject


class LocalComputeConfig(DataObject):
    pass


class VMConfig(DataObject):
    pass


class AwsVMConfig(VMConfig):
    pass


class GcpVMConfig(VMConfig):
    pass


class AzureVMConfig(VMConfig):
    pass


class LightningAIStudioConfig(DataObject):
    pass


class ComputeConfig(DataObject):
    local: Optional[LocalComputeConfig] = None
    vm: Optional[VMConfig] = None
    kubernetes: Optional[KubernetesConfig] = None
    lightning_ai_studio: Optional[LightningAIStudioConfig] = None
    groq: Optional[GroqConfig] = None
