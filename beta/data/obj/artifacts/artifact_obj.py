from __future__ import annotations

import typing
if typing.TYPE_CHECKING:
    pass

import pyarrow as pa
import gzip
import hashlib
import mimetypes
import uuid
import ulid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import Field, HttpUrl

from beta.data.obj.data_object import DataObject

class ArtifactError(Exception):
    """Base exception class for Artifact errors."""

class PayloadError(ArtifactError):
    """Raised when there's an issue with the artifact payload."""


class Artifact(DataObject):
    """
    Versatile model for managing artifacts in a data system.
    ─────────────────────────────────────────────
    Attributes:                                 
      id               │ ULID identifier         
      name             │ Artifact name           
      artifact_uri     │ Storage URI             
      payload          │ Gzipped content         
      extension        │ File extension          
      mime_type        │ MIME type               
      version          │ Artifact version        
      size_bytes       │ Size in bytes           
      checksum         │ MD5 checksum            
      account_id       │ Account UUID            
      user_id          │ User UUID               
      user_session_id  │ Session ULID            
      region_id        │ Geographic region       
      is_ai_generated  │ AI generation flag      
      sensitivity_level│ Confidentiality level   
      context_event_id │ Context event ULID      
      updated_at       │ Last update timestamp   
      metadata         │ Additional metadata     
    ─────────────────────────────────────────────
    Methods:                                    
      compress_payload(payload: bytes) -> bytes  
        Compress the given payload using gzip    
      decompress_payload(payload: bytes) -> bytes
        Decompress the given gzipped payload     
      verify_integrity() -> bool                 
        Verify artifact integrity via checksum   
      from_file(file_path: str, **kwargs)        
        Create Artifact instance from file       
    ──��──────────────────────────────────────────
    Validators:                                 
      set_mime_type()  │ Auto-set MIME type      
      set_checksum()   │ Compute MD5 checksum    
    ─────────────────────────────────────────────
    Usage:                                      
      session = get_session()                                                                                                                                                                                                               │
      artifact = Artifact.from_file(             
        "/path/to/file.pdf",                     
        user_session = session,                  
        is_ai_generated = False,                 
        sensitivity_level = 'CONFIDENTIAL'       
      )                                          
      is_valid = artifact.verify()               
    ─────────────────────────────────────────────
    Notes:                                      
      • Inherits from CustomModel                
      • Payload always stored gzipped            
      • Handle PayloadError when using payload   
    ─────────────────────────────────────────────
    """
    def __init__(self,
        name: str = Field(
            ...,
            description="Name of the artifact",
            example="Q2 Financial Report",
            max_length=255
        ),
        payload: Optional[bytes] = Field(
            None,
            description="The artifact object payload, always gzipped",
            example=b"[gzipped content]"
        ),
        extension: str = Field(
            ...,
            description="File extension of the artifact.",
            example="pdf"
        ),
        mime_type: str = Field(
            ...,
            description="MIME type of the artifact",
            example="application/pdf"
        ),
        version: int = Field(
            default=1,
            description="Version of the artifact",
            example=1
        ),
        size_bytes: int = Field(
            ...,
            description="Size of the artifact in bytes",
            example=1048576
        ),
        checksum: str = Field(
            ...,
            description="MD5 checksum of the artifact payload",
            example="d41d8cd98f00b204e9800998ecf8427e"
        )
        session_id: ulid.ULID = Field(
            ...,
            description="Unique identifier of the user session during which this artifact was created or last modified",
            example="abcdef12-3456-7890-abcd-ef1234567890"
        ),
        account_id: uuid.UUID = Field(
            ...,
            description="Unique identifier of the account associated with this artifact",
            example="98765432-1234-5678-9abc-defghijklmno"
        ),
        user_id: uuid.UUID = Field(
            ...,
            description="Unique identifier of the user who created or last modified this artifact",
            example="12345678-abcd-efgh-ijkl-mnopqrstuvwx"
        ),
        region_id: str = Field(
            ...,
            description="Identifier of the geographic region associated with this artifact",
            example="us-west-2",
            max_length=50
        ),
        is_ai_generated: bool = Field(
            ...,
            description="Indicates whether the artifact was generated by an AI system",
            example=False
        ),
        context_event_id: str = Field(
            default_factory=lambda: ulid.new().str,
            description="ULID Unique identifier for the corresponding context event that triggered the upload of the artifact.",
            example="01F8MECHZCP3QV1Z4T5BA4WJXN"
        )
    ):
    self.artifact_uri: HttpUrl = Field(
        ...,
        description="URI where the artifact is stored",
        example="gs://your-bucket/artifacts/account_id/user_id/q2_financial_report.pdf.gz"
    )

    @validator('mime_type', pre=True, always=True)
    def set_mime_type(cls, v, values):
        if v:
            return v
        ext = values.get('extension')
        if ext:
            mime_type, _ = mimetypes.guess_type(f"file.{ext}")
            if mime_type:
                return mime_type
        raise ValueError("Unable to determine MIME type")

    @validator('checksum', pre=True, always=True)
    def set_checksum(cls, v, values):
        if v:
            return v
        payload = values.get('payload')
        if payload:
            return hashlib.md5(payload).hexdigest()
        raise ValueError("Unable to compute checksum: no payload provided")
    
    def persist(self):
        # Write Payload to Object Store

    
    def __repr__(self):
        return f"<Artifact id={self.id} {self.name}.{self.extension} ({self.mime_type})>"

    def __str__(self):
        return self.__repr__()


