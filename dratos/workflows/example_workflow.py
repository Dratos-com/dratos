from dratos.agents.agent import Agent
from dratos.helpers.schema_validator import SchemaValidator, MetadataSchema
from dratos.helpers.serializer import Serializer, DeltaCatBackend
from dratos.data.obj.result import Result

def main():
    # Initialize Schema Validator
    schemas = {"metadata_schema": MetadataSchema}
    validator = SchemaValidator(schemas=schemas)

    # Initialize Serializer with DeltaCat Backend
    backend = DeltaCatBackend()
    serializer = Serializer(backend=backend)

    # Create Metadata
    metadata = Metadata(schema="metadata_schema", provenance="system")

    # Initialize Agent
    agent = Agent(name="ProcessorAgent", models=[], tools=[], metadata=metadata)

    # Execute Pipeline
    input_data = {"key": "value"}
    result = agent.execute_pipeline(input_data)

    if result.is_error:
        print(f"Workflow failed with error: {result.value}")
    else:
        print(f"Workflow succeeded with result: {result.value}")

if __name__ == "__main__":
    main()