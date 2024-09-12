from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    pass


import ray
from typing import Optional, List
from deltacat import Namespace, Table, Partition, Delta
from api.session import Session
from beta.services.unity_catalog import UnityCatalogService


@ray.remote
class Data:
    def __init__(self, config):
        self.config = config
        self.artifacts = ArtifactManager.remote(config)
        self.documents = DocumentsManager.remote(config)
        self.graphs = GraphsManager.remote(config)


@ray.remote
class ArtifactManager:
    def __init__(self, config):
        self.config = config
        self.unity_catalog = UnityCatalogService(config)
        self.accessor = ArtifactAccessor.remote(config)
        self.selector = ArtifactSelector.remote(config)

    async def search(self, query: str, session: Optional[Session] = None) -> List[dict]:
        # Implement artifact search logic here using Unity Catalog and DeltaCat
        results = await self.selector.search.remote(query)
        if session:
            session.data["last_artifact_search"] = query
            await self.config.update_session(session)
        return results


@ray.remote
class ArtifactAccessor:
    def __init__(self, config):
        self.config = config
        self.unity_catalog = UnityCatalogService(config)

    async def get_artifact(
        self, namespace: str, table: str, partition: Optional[str] = None
    ) -> Delta:
        ns = Namespace(namespace)
        tbl = Table(ns, table)
        if partition:
            part = Partition(tbl, partition)
            return await self.unity_catalog.get_delta(part)
        else:
            return await self.unity_catalog.get_latest_delta(tbl)

    async def upsert_artifact(
        self, namespace: str, table: str, data: dict, partition: Optional[str] = None
    ) -> None:
        ns = Namespace(namespace)
        tbl = Table(ns, table)
        if partition:
            part = Partition(tbl, partition)
            await self.unity_catalog.upsert_delta(part, data)
        else:
            await self.unity_catalog.upsert_delta(tbl, data)


@ray.remote
class ArtifactSelector:
    def __init__(self, config):
        self.config = config
        self.unity_catalog = UnityCatalogService(config)

    async def search(self, query: str) -> List[dict]:
        # Implement search logic using Unity Catalog and DeltaCat
        results = await self.unity_catalog.search(query)
        return [self._format_result(r) for r in results]

    def _format_result(self, result: dict) -> dict:
        # Format the search result
        return {
            "namespace": result.get("namespace"),
            "table": result.get("table"),
            "partition": result.get("partition"),
            "metadata": result.get("metadata", {}),
        }


class ArtifactFactory:
    @staticmethod
    def create(config, artifact_type: str):
        if artifact_type == "document":
            return DocumentArtifact.remote(config)
        elif artifact_type == "graph":
            return GraphArtifact.remote(config)
        else:
            raise ValueError(f"Unknown artifact type: {artifact_type}")


@ray.remote
class DocumentArtifact:
    def __init__(self, config):
        self.config = config
        # Initialize document-specific properties


@ray.remote
class GraphArtifact:
    def __init__(self, config):
        self.config = config
        # Initialize graph-specific properties


class Data:
    def __init__(self, config):
        self.config = config
        self.artifacts = ArtifactManager(config)
        self.documents = DocumentsManager(config)
        self.graphs = GraphsManager(config)


class ArtifactManager:
    def __init__(self, config):
        self.config = config

    def search(self, query):
        # Implement artifact search logic here
        # This could involve querying MLflow, Unity Catalog, or other artifact stores
        pass


class ArtifactAccessor:
    pass


class ArtifactSelector:
    pass


class ArtifactFactory:
    pass
