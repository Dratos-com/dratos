from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    pass
from ...schema.base import base
import pyarrow as pa

# Node schema
node = pa.schema(
    [
        ("id", pa.string()),
        ("label", pa.string()),
        ("properties", pa.struct([("type", pa.string()), ("value", pa.string())])),
    ]
).append(base)

# Edge schema
edge = pa.schema(
    [
        ("id", pa.string()),
        ("source", pa.string()),
        ("target", pa.string()),
        ("label", pa.string()),
        ("properties", pa.struct([("type", pa.string()), ("value", pa.string())])),
    ]
).append(base)

# Graph schema
graph = pa.schema(
    [
        ("id", pa.string()),
        ("name", pa.string()),
        ("description", pa.string()),
        ("nodes", pa.list_(node)),
        ("edges", pa.list_(edge)),
        ("metadata", pa.map_(pa.string(), pa.string())),
    ]
).append(base)


# Create a GraphManager class for easier manipulation
class GraphManager:
    def __init__(self):
        self.graph_schema = graph
        self.node_schema = node
        self.edge_schema = edge

    def create_node(self, id, label, properties):
        return pa.RecordBatch.from_arrays(
            [pa.array([id]), pa.array([label]), pa.array([properties])],
            schema=self.node_schema,
        )

    def create_edge(self, id, source, target, label, properties):
        return pa.RecordBatch.from_arrays(
            [
                pa.array([id]),
                pa.array([source]),
                pa.array([target]),
                pa.array([label]),
                pa.array([properties]),
            ],
            schema=self.edge_schema,
        )

    def create_graph(self, id, name, description, nodes, edges, metadata):
        return pa.RecordBatch.from_arrays(
            [
                pa.array([id]),
                pa.array([name]),
                pa.array([description]),
                pa.array([nodes]),
                pa.array([edges]),
                pa.array([metadata]),
            ],
            schema=self.graph_schema,
        )


# Instantiate the GraphManager
graph_manager = GraphManager()


import spacy
from deltacat import Writer
from iceberg.pyiceberg import Schema, Types
from iceberg.pyiceberg.catalog import load_catalog

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Define Iceberg schema for dependency parsing results
dependency_schema = Schema(
    Types.NestedField(1, "text", Types.StringType()),
    Types.NestedField(2, "tokens", Types.ListType(Types.StringType())),
    Types.NestedField(
        3,
        "dependencies",
        Types.ListType(
            Types.StructType(
                Types.NestedField(1, "token", Types.StringType()),
                Types.NestedField(2, "dep", Types.StringType()),
                Types.NestedField(3, "head", Types.StringType()),
            )
        ),
    ),
)


def process_text(text):
    doc = nlp(text)
    tokens = [token.text for token in doc]
    dependencies = [
        {"token": token.text, "dep": token.dep_, "head": token.head.text}
        for token in doc
    ]
    return {"text": text, "tokens": tokens, "dependencies": dependencies}


def save_to_iceberg(data, catalog_name, database_name, table_name):
    # Load Unity Catalog
    catalog = load_catalog(catalog_name)

    # Create or get the table
    try:
        table = catalog.load_table(f"{database_name}.{table_name}")
    except:
        table = catalog.create_table(
            f"{database_name}.{table_name}", schema=dependency_schema
        )

    # Write data using DeltaCat
    with Writer(table) as writer:
        writer.write(data)


def text_pipeline(texts):
    processed_data = [process_text(text) for text in texts]
    save_to_iceberg(
        processed_data, "unity_catalog", "my_database", "dependency_parsing_results"
    )


# Example usage
# texts = ["This is a sample sentence.", "Another example for parsing."]
# text_pipeline(texts)


# Example usage
# client = serve.get_deployment("DependencyParsingService").get_handle()
# texts = ["This is a sample sentence.", "Another example for parsing."]
# ray.get(client.remote(texts))
