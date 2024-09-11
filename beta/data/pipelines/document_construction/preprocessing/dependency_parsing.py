from __future__ import annotations

import typing
if typing.TYPE_CHECKING:
    pass
import ray
from ray import serve
from typing import List
from beta.agents.base import Agent
from beta.tasks.base import Task
from beta.environment.base import Environment


@ray.remote
class DependencyParsingAgent(Agent):
    def __init__(self):
        super().__init__()
        self.nlp = spacy.load("en_core_web_sm")

    def process_text(self, text: str) -> dict:
        doc = self.nlp(text)
        tokens = [token.text for token in doc]
        dependencies = [
            {"token": token.text, "dep": token.dep_, "head": token.head.text}
            for token in doc
        ]
        return {"text": text, "tokens": tokens, "dependencies": dependencies}


class DependencyParsingTask(Task):
    def __init__(self, texts: List[str]):
        super().__init__()
        self.texts = texts

    async def execute(self, agent: DependencyParsingAgent) -> List[dict]:
        return await agent.process_text.remote(self.texts)


class IcebergEnvironment(Environment):
    def __init__(self, catalog_name: str, database_name: str, table_name: str):
        super().__init__()
        self.catalog_name = catalog_name
        self.database_name = database_name
        self.table_name = table_name

    def save_to_iceberg(self, data: List[dict]):
        catalog = load_catalog(self.catalog_name)

        try:
            table = catalog.load_table(f"{self.database_name}.{self.table_name}")
        except:
            table = catalog.create_table(
                f"{self.database_name}.{self.table_name}", schema=dependency_schema
            )

        with Writer(table) as writer:
            writer.write(data)


@serve.deployment
class DependencyParsingService:
    def __init__(self):
        self.agent = DependencyParsingAgent.remote()
        self.environment = IcebergEnvironment(
            "unity_catalog", "my_database", "dependency_parsing_results"
        )

    async def __call__(self, texts: List[str]) -> str:
        task = DependencyParsingTask(texts)
        results = await task.execute(self.agent)
        self.environment.save_to_iceberg(results)
        return "Processing complete and results saved to Iceberg."


# Deploy the service
deployment = DependencyParsingService.bind()
import daft
from daft.udf import sql_udf
import spacy
import cupy as cp

# Load spaCy model with GPU support
nlp = spacy.load("en_core_web_sm")
if spacy.prefer_gpu():
    nlp.to_gpu()


@sql_udf(
    return_dtype=daft.DataType.struct(
        [
            ("text", daft.DataType.string()),
            ("tokens", daft.DataType.list(daft.DataType.string())),
            (
                "dependencies",
                daft.DataType.list(
                    daft.DataType.struct(
                        [
                            ("token", daft.DataType.string()),
                            ("dep", daft.DataType.string()),
                            ("head", daft.DataType.string()),
                        ]
                    )
                ),
            ),
        ]
    )
)
def dependency_parse(text: str):
    doc = nlp(text)
    tokens = cp.asnumpy(doc.to_array([spacy.attrs.ORTH])).flatten().tolist()
    dependencies = [
        {"token": token.text, "dep": token.dep_, "head": token.head.text}
        for token in doc
    ]
    return {"text": text, "tokens": tokens, "dependencies": dependencies}


class DaftDependencyParsingTask(Task):
    def __init__(self, texts: List[str]):
        super().__init__()
        self.texts = texts

    async def execute(self) -> daft.DataFrame:
        df = daft.from_pylist([{"text": text} for text in self.texts])
        return df.select(dependency_parse("text").alias("parsed")).repartition(
            num_partitions=8
        )


class DaftIcebergEnvironment(Environment):
    def __init__(self, catalog_name: str, database_name: str, table_name: str):
        super().__init__()
        self.catalog_name = catalog_name
        self.database_name = database_name
        self.table_name = table_name

    def save_to_iceberg(self, df: daft.DataFrame):
        df.write_iceberg(
            table_name=f"{self.catalog_name}.{self.database_name}.{self.table_name}",
            mode="overwrite",
        )


@serve.deployment(num_gpus=1)
class DaftDependencyParsingService:
    def __init__(self):
        self.environment = DaftIcebergEnvironment(
            "unity_catalog", "my_database", "dependency_parsing_results"
        )

    async def __call__(self, texts: List[str]) -> str:
        task = DaftDependencyParsingTask(texts)
        results = await task.execute()
        self.environment.save_to_iceberg(results)
        return "Processing complete and results saved to Iceberg using Daft with CUDA optimization."


# Deploy the new Daft-based service with GPU support
daft_deployment = DaftDependencyParsingService.bind()
