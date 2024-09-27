from enum import Enum

from typing import Optional, Any, List, Dict
from dratos.data.obj.base.data_object import DataObject
import daft


class ContentType(Enum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    SIGNAL = "signal"
    CODE = "code"
    THREE_D = "3d"
    HTML = "html"
    JSON = "json"
    XML = "xml"
    YAML = "yaml"
    CSV = "csv"
    SQL = "sql"
    PDF = "pdf"
    MD = "md"
    OTHER = "other"
    URL = "url"
    BINARY = "binary"


class Node(DataObject):
    def __init__(
        self,
        content: Optional[Any],
        content_type: ContentType,
        embeddings: Optional[daft.DataType.embeddings],
        properties: Optional[Dict[str, Any]],
    ):
        super().__init__()
        self.content = content
        self.content_type = content_type
        self.embeddings = embeddings
        self.properties = properties

    def __repr__(self):
        return f"Node(id={self.id}, content_type={self.content_type}, content={self.content}, embeddings={self.embeddings}, properties={self.properties})"

    def __str__(self):
        return self.__repr__()

    # he content_type should be intelligently inferred from the content.
    def infer_content_type(self):
        if isinstance(self.content, str):
            if self.content.startswith("http"):
                return ContentType.URL
            elif self.content.endswith(
                (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp")
            ):
                return ContentType.IMAGE
            elif self.content.endswith(
                (".mp4", ".mov", ".avi", ".mkv", ".flv", ".wmv")
            ):
                return ContentType.VIDEO
            elif self.content.endswith(
                (".wav", ".mp3", ".ogg", ".flac", ".aiff", ".m4a")
            ):
                return ContentType.AUDIO
            elif self.content.endswith(
                (
                    ".py",
                    ".js",
                    "jsx",
                    "ts",
                    "tsx",
                    ".html",
                    ".css",
                    ".sql",
                    ".rb",
                    ".php",
                    ".go",
                    ".rs",
                    ".swift",
                    ".kt",
                    ".java",
                    ".c",
                    ".cpp",
                    ".h",
                    ".hpp",
                    ".md",
                    ".ini",
                    ".cfg",
                    ".conf",
                    ".props",
                    ".settings",
                    ".properties",
                    ".env",
                    ".env.local",
                    ".env.development",
                    ".env.production",
                    ".env.test",
                    ".env.staging",
                    ".env.development.local",
                    ".env.production.local",
                    ".env.test.local",
                    ".env.staging.local",
                    ".sh",
                )
            ):
                return ContentType.CODE
            else:
                return ContentType.TEXT
        elif isinstance(self.content, bytes):
            return ContentType.BINARY
        else:
            return ContentType.OTHER

    def get_embedding_dimensions(self):
        if self.embeddings:
            return self.embeddings.dimensions
        else:
            return None
