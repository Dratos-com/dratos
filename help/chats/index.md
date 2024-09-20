Index of Foundational Primitives
Document
Artifact
Graph
Node
Edge
Metadata
Inference Engine
Serving Engine
Schema Validator
Dataframe Adapter
Model
Tool
Agent
Grammar
EmbeddingType (Daft Advanced Data Type)
ImageType (Daft Advanced Data Type)
URLType (Daft Advanced Data Type)
Serializer (DeltaCat Integration)
README for Foundational Primitives
Overview
This document outlines the core primitives that form the foundation of our composable agent-based system, integrating advanced data processing capabilities using Daft and Apache Arrow, with DeltaCat providing streaming and serialization capabilities. These primitives are designed to handle both structured and unstructured data, serving as the building blocks for complex systems in a highly modular and flexible manner.

The system enables users to define agents, models, tools, and systems, combining them into cohesive grammars while preserving data quality and integrity through schema validation and native support for advanced data types.

1. Document
Description: A Document is a structured data object consisting of multiple Node objects. Each document is associated with relevant metadata.
Attributes:
nodes: List of Node objects.
metadata: Instance of Metadata.
Methods:
validate(): Validates the document structure against a schema.
serialize(): Converts the document to Arrow format for storage in DeltaCat.
2. Artifact
Description: An Artifact is a data object that represents a non-document resource, such as a file or image, with associated metadata.
Attributes:
file: Path or reference to the resource.
metadata: Instance of Metadata.
Methods:
validate(): Ensures artifact structure complies with metadata constraints.
serialize(): Converts the artifact into an Arrow-compatible format.
3. Graph
Description: A Graph is a structured object that consists of Node and Edge objects. It can represent relationships and flows within a system.
Attributes:
nodes: List of Node objects.
edges: List of Edge objects.
metadata: Instance of Metadata.
Methods:
validate(): Ensures graph nodes and edges follow defined schema.
serialize(): Converts the graph to Arrow format for serialization.
4. Node
Description: A Node represents a single element in a document or graph. Each node has attributes and metadata that provide additional context.
Attributes:
data: Content of the node.
metadata: Instance of Metadata.
Methods:
validate(): Ensures the node conforms to a schema.
serialize(): Converts the node data into Arrow format.
5. Edge
Description: An Edge represents a relationship between two Node objects in a graph. It defines the direction and context of the relationship.
Attributes:
start_node: Reference to the starting Node.
end_node: Reference to the ending Node.
metadata: Instance of Metadata.
6. Metadata
Description: Metadata encapsulates important information about a document, graph, node, or artifact, such as provenance, timestamp, and quality indicators.
Attributes:
schema: Reference to the schema used for validation.
timestamp: Time of creation or modification.
provenance: Origin or source of the data.
Methods:
validate(): Ensures metadata integrity against the schema.
7. Inference Engine
Description: An Inference Engine performs model inference. It is an adapter for serving engines that vary based on the customer's device or environment.
Attributes:
engine_type: Type of serving engine (e.g., TensorFlow, ONNX, etc.).
model: Associated Model to run inference on.
Methods:
run_inference(input_data): Runs inference on the given data.
Serving Engine (Subclass of Inference Engine)
Description: A specialized Inference Engine that handles serving models on different devices.
Attributes:
device: Device-specific serving configuration.
adapter: Adapter that integrates with the customer's existing infrastructure.
8. Schema Validator
Description: A Schema Validator ensures that documents, graphs, nodes, and artifacts conform to predefined schemas.
Methods:
validate(object): Validates an object against its schema.
9. Dataframe Adapter
Description: The Dataframe Adapter facilitates the conversion between domain objects (documents, graphs, etc.) and Daft dataframes for processing and querying.
Methods:
to_dataframe(object): Converts a domain object into a Daft dataframe.
from_dataframe(df): Converts a Daft dataframe back into a domain object.
10. Model
Description: A Model is a representation of a machine learning model, which can be used by the inference engine for predictions.
Attributes:
architecture: Model architecture (e.g., Transformer, CNN).
parameters: Model parameters.
metadata: Instance of Metadata.
11. Tool
Description: A Tool is a utility or resource that supports the processing of data or models. Tools are designed to work with Daft and Arrow for efficient processing.
Attributes:
type: Type of tool (e.g., preprocessor, postprocessor).
functionality: Core functionality provided by the tool.
12. Agent
Description: An Agent is a container for models and tools. It orchestrates the execution of models and processing pipelines.
Attributes:
models: List of associated Model objects.
tools: List of associated Tool objects.
metadata: Instance of Metadata.
13. Grammar
Description: A Grammar defines the structure and rules for combining models and tools into larger systems.
Attributes:
rules: Set of rules defining the system's structure.
components: List of models, tools, and agents involved.
14. EmbeddingType (Daft Advanced Data Type)
Description: A specialized data type for embeddings, designed for Daft dataframes. Supports efficient storage and querying.
Attributes:
embedding_vector: Vector representation of data.
15. ImageType (Daft Advanced Data Type)
Description: A specialized data type for images, designed for Daft dataframes. Supports efficient processing and transformation.
Attributes:
image_data: Binary or array-based representation of the image.
16. URLType (Daft Advanced Data Type)
Description: A specialized data type for URLs, designed for Daft dataframes. Supports efficient retrieval and storage of web resources.
Attributes:
url: String representation of the URL.
17. Serializer (DeltaCat Integration)
Description: The Serializer handles the conversion of domain objects (e.g., documents, artifacts, graphs) into Arrow format for serialization and integration with DeltaCat.
Methods:
serialize(object): Serializes the object for storage.
upsert(object): Performs upserts using DeltaCat in a streaming use case.