class JsonDataObject:
    def __init__(self, data: Union[DataFrame, pa.Table]):
        self.data = data

    def to_daft(self) -> DataFrame:
        # Convert to Daft DataFrame
        pass

    def to_arrow(self) -> pa.Table:
        # Convert to Arrow Table
        pass
