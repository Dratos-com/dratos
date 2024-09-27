# Project Name

This project implements a flexible and efficient machine learning serving system with support for various language models and engines.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Key Abstractions](#key-abstractions)
- [Components](#components)
  - [Engines](#engines)
  - [Models](#models)
  - [Tools](#tools)
  - [Data Management](#data-management)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project provides a comprehensive framework for serving machine learning models, particularly focusing on language models. It offers support for various engines, including OpenAI, VLLM, Transformers, LlamaCpp, and Triton Server. The system is designed to be flexible, allowing easy integration of new models and engines.

## Project Structure

The project is organized into several key directories:

- `dratos/`: Core implementation of the project
  - `models/`: Model implementations and serving logic
  - `data/`: Data management and processing
  - `tools/`: Utility tools for various tasks
  - `agents/`: Agent implementations for task execution
- `api/`: API-related code, including deployments
- `examples/`: Example notebooks and usage demonstrations

## Setup

1. Install the required dependencies:

```
pip install -r requirements.txt
```

2. Set up the necessary environment variables, including API keys for various services (e.g., OpenAI, VLLM) in `.env.local`. You can use `.env.example` as a template.

3. Initialize the Ray cluster if using distributed computing features.

## Usage

Here's a basic example of how to use the system:

```python
from dratos import Agents, Models, Tools, Data
import mlflow

# Initialize the mlflow tracking server


# Initialize the engine
engine = VLLMEngine(
  model_name="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO", mlflow_client=mlflow_client
)

# Get the model
model = Models.get(
  model_name="gpt-3.5-turbo",
  engine=engine,
  mlflow_client=mlflow_client
)

# Create an agent
agent = Agent(engine=engine)

# Generate text
response = agent.generate("Tell me a joke about programming.")
print(response)
```

## Key Abstractions

This project uses several key abstractions to provide a flexible and extensible architecture:

### 1. Engine

The `Engine` abstraction represents different backend systems for model execution. Key implementations include:

- `OpenAIEngine`: Interfaces with OpenAI's API
- `VLLMEngine`: Uses the VLLM library for efficient language model inference
- `TransformersEngine`: Utilizes Hugging Face's Transformers library
- `LlamaCppEngine`: Integrates with the Llama.cpp library
- `TritonServerEngine`: Supports NVIDIA's Triton Inference Server

All engines implement a common interface, allowing for easy swapping and integration of new engines.

### 2. Model

The `Model` abstraction represents different types of machine learning models. Key types include:

- `LanguageModel`: For text generation and processing
- `SpeechToTextModel`: For transcription tasks
- `StructuredGenerationModel`: For generating structured outputs

Models are implemented as subclasses of `BaseModel` and can be extended to support new model types.

### 3. Tool

The `Tool` abstraction represents utility functions that can be used by agents. Examples include:

- `CalculatorTool`: For performing mathematical calculations
- `DataframeTool`: For working with structured data
- `KnowledgeBaseTool`: For managing and querying a knowledge base

Tools are implemented as subclasses of `BaseTool` and can be easily extended.

### 4. Agent

The `Agent` abstraction represents an entity that can use models and tools to perform tasks. Agents can be configured with different models and sets of tools to handle various tasks.

### 5. DataObject

The `DataObject` abstraction is a base class for all data objects in the system. It provides common fields and methods for data manipulation, including conversion to and from Arrow format.

### 6. Task

The `Task` abstraction represents a unit of work to be performed by an agent. Tasks can be chained together to form complex workflows.

### 7. Environment

The `Environment` abstraction represents the context in which agents operate. It can include configuration settings, available resources, and constraints.

### 8. Prompt

The `Prompt` abstraction represents a template for generating text prompts. It uses Jinja2 for template rendering and supports dynamic prompt generation based on input parameters.

### 9. KnowledgeBase

The `KnowledgeBase` abstraction represents a collection of information that can be queried and updated by agents and tools.

### 10. Deployment

The `Deployment` abstraction represents a served model or application. It handles the lifecycle of deploying and managing models in production environments.

These abstractions work together to create a flexible and powerful system for AI model serving and task execution. The modular design allows for easy extension and customization to meet specific use cases.

## Components

### Engines

This project supports multiple engines for serving models. The current supported engines are:


For more detailed examples, refer to the notebooks in the `examples/` directory.

## Components

### Engines

The project supports various engines for model serving:

- OpenAI Engine: Interfaces with OpenAI's API
- VLLM Engine: Uses the VLLM library for efficient language model inference
- Transformers Engine: Utilizes Hugging Face's Transformers library
- LlamaCpp Engine: Integrates with the Llama.cpp library
- Triton Server Engine: Supports NVIDIA's Triton Inference Server

Each engine implements a common interface, allowing for easy swapping and integration of new engines.

### Models

The system supports various types of models, including:

- Language Models: For text generation and processing
- Speech-to-Text Models: For transcription tasks
- Structured Generation Models: For generating structured outputs

Models are implemented as subclasses of `BaseModel` and can be easily extended to support new model types.

### Tools

The project includes a set of utility tools that can be used by agents:

- Calculator Tool: For performing mathematical calculations
- Dataframe Tool: For working with structured data
- Knowledge Base Tool: For managing and querying a knowledge base

Tools are implemented as subclasses of `BaseTool` and can be easily extended to add new functionalities.

### Data Management

The project includes a robust data management system:

- Artifact Management: For handling various types of artifacts
- Document Management: For processing and storing textual documents
- Graph Management: For working with knowledge graphs

Data is managed using Ray for distributed computing and DeltaCat for efficient data storage and retrieval.

## Configuration

Configuration is managed through a combination of environment variables and configuration files. Key configuration options include:

- Model selection and parameters
- Engine-specific settings
- API keys and endpoints
- Data storage locations

Refer to the `config/` directory for detailed configuration options.

## Contributing

Contributions to the project are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch for your feature
3. Implement your changes
4. Write tests for your new functionality
5. Submit a pull request

Please ensure your code adheres to the project's coding standards and includes appropriate documentation.

## License

Apache 2.0