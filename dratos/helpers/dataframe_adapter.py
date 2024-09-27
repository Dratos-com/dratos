from abc import ABC, abstractmethod
from ..data.obj.result import Result
import daft

class DataframeHandler(ABC):
    @abstractmethod
    def to_dataframe(self, obj) -> Result[daft.DataFrame, Exception]:
        pass

    @abstractmethod
    def from_dataframe(self, df: daft.DataFrame) -> Result[Any, Exception]:
        pass

class DaftHandler(DataframeHandler):
    def to_dataframe(self, obj) -> Result[daft.DataFrame, Exception]:
        try:
            df = daft.from_pydict(obj.__dict__)  # Simplified example
            return Result.Ok(df)
        except Exception as e:
            return Result.Error(e)

    def from_dataframe(self, df: daft.DataFrame) -> Result[Any, Exception]:
        try:
            obj = SomeDomainObject(**df.to_pydict())  # Replace with actual logic
            return Result.Ok(obj)
        except Exception as e:
            return Result.Error(e)

class DataframeAdapter:
    def __init__(self, handler: DataframeHandler):
        self.handler = handler

    def to_dataframe(self, obj) -> Result[daft.DataFrame, Exception]:
        return self.handler.to_dataframe(obj)

    def from_dataframe(self, df: daft.DataFrame) -> Result[Any, Exception]:
        return self.handler.from_dataframe(df)