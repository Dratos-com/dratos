from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    pass

from dratos.data.obj import DataObject
import pyarrow as pa
import deltacat
import daft
from typing import Union


class JsonDataObject(DataObject):
    def __init__(self, data: Union[DataFrame, pa.Table]):
        self.data = data

    def to_daft(self) -> DataFrame:
        # Convert to Daft DataFrame
        pass

    def to_arrow(self) -> pa.Table:
        # Convert to Arrow Table
        pass
