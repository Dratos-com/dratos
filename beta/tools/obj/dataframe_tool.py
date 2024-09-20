from __future__ import annotations

import typing
from typing import Any, Dict, List, Optional, Union

if typing.TYPE_CHECKING:
    pass

import daft
import ray

@ray.remote
class DataFrameTool():
    """
    DataFrameTool provides a comprehensive API for working with Daft DataFrames.

    This tool allows users and LLMs to perform various operations on Local/Remote Daft DataFrames,
    including data creation, manipulation, filtering, reordering, combining, aggregation, execution,
    writing, and schema/lineage operations.

    Key features:
    1. Data Creation: Read from CSV, Parquet, JSON, or create from Python data structures.
    2. Data Manipulation: Select columns, add/modify columns, pivot/unpivot, explode, and transform data.
    3. Filtering: Apply distinct, where, limit, and sampling operations.
    4. Reordering: Sort data and manage partitions.
    5. Combining: Join and concatenate DataFrames.
    6. Aggregation: Group by, aggregate, and perform common statistical operations.
    7. Execution: Collect results, show data, and convert to other formats (Pandas, Arrow, Python dict).
    8. Writing: Save data to Parquet or CSV formats.
    9. Schema and Lineage: Explain operations, retrieve schema, and get column names.

    Each method in this class corresponds to a Daft DataFrame operation, making it easy to perform
    complex data manipulations and analyses through a simple, intuitive interface.

    Usage:
    Instantiate this tool and call its methods, passing Daft DataFrames and other required parameters
    to perform the desired operations on your data.
    """

    name = "dataframe"
    description = "Performs operations on Local/Remote Daft DataFrames"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.daft = config.get_daft()

    # Data Creation Methods
    def read_csv(self, path: Union[str, List[str]], **kwargs) -> daft.DataFrame:
        return self.daft.read_csv(path, **kwargs)

    def read_parquet(self, path: Union[str, List[str]], **kwargs) -> daft.DataFrame:
        return self.daft.read_parquet(path, **kwargs)

    def read_json(self, path: Union[str, List[str]], **kwargs) -> daft.DataFrame:
        return self.daft.read_json(path, **kwargs)

    def from_pydict(self, data: Dict[str, Any]) -> daft.DataFrame:
        return self.daft.from_pydict(data)

    def from_pylist(self, data: List[Dict[str, Any]]) -> daft.DataFrame:
        return self.daft.from_pylist(data)

    def from_arrow(self, data: Union[Any, List[Any]]) -> daft.DataFrame:
        return self.daft.from_arrow(data)

    def from_pandas(self, data: Union[Any, List[Any]]) -> daft.DataFrame:
        return self.daft.from_pandas(data)

    # Data Manipulation Methods
    def select(self, df: daft.DataFrame, *exprs) -> daft.DataFrame:
        return df.select(*exprs)

    def with_column(self, df: daft.DataFrame, name: str, expr) -> daft.DataFrame:
        return df.with_column(name, expr)

    def with_columns(self, df: daft.DataFrame, **exprs) -> daft.DataFrame:
        return df.with_columns(**exprs)

    def pivot(
        self, df: daft.DataFrame, index: List[str], columns: str, values: str
    ) -> daft.DataFrame:
        return df.pivot(index, columns, values)

    def exclude(self, df: daft.DataFrame, *columns: str) -> daft.DataFrame:
        return df.exclude(*columns)

    def explode(self, df: daft.DataFrame, column: str) -> daft.DataFrame:
        return df.explode(column)

    def unpivot(
        self,
        df: daft.DataFrame,
        id_vars: List[str],
        value_vars: List[str],
        var_name: str,
        value_name: str,
    ) -> daft.DataFrame:
        return df.unpivot(id_vars, value_vars, var_name, value_name)

    def transform(self, df: daft.DataFrame, func) -> daft.DataFrame:
        return df.transform(func)

    # Filtering Methods
    def distinct(self, df: daft.DataFrame) -> daft.DataFrame:
        return df.distinct()

    def where(self, df: daft.DataFrame, condition) -> daft.DataFrame:
        return df.where(condition)

    def limit(self, df: daft.DataFrame, n: int) -> daft.DataFrame:
        return df.limit(n)

    def sample(
        self, df: daft.DataFrame, fraction: float, seed: Optional[int] = None
    ) -> daft.DataFrame:
        return df.sample(fraction, seed=seed)

    # Reordering Methods
    def sort(self, df: daft.DataFrame, by, ascending: bool = True) -> daft.DataFrame:
        return df.sort(by, ascending=ascending)

    def repartition(self, df: daft.DataFrame, num_partitions: int) -> daft.DataFrame:
        return df.repartition(num_partitions)

    def into_partitions(
        self, df: daft.DataFrame, num_partitions: int
    ) -> daft.DataFrame:
        return df.into_partitions(num_partitions)

    # Combining Methods
    def join(
        self, df: daft.DataFrame, other: daft.DataFrame, on, how: str = "inner"
    ) -> daft.DataFrame:
        return df.join(other, on=on, how=how)

    def concat(self, df: daft.DataFrame, other: daft.DataFrame) -> daft.DataFrame:
        return df.concat(other)

    # Aggregation Methods
    def groupby(self, df: daft.DataFrame, by) -> daft.GroupBy:
        return df.groupby(by)

    def agg(self, df: daft.DataFrame, *args, **kwargs) -> daft.DataFrame:
        return df.agg(*args, **kwargs)

    def sum(self, df: daft.DataFrame, *columns) -> daft.DataFrame:
        return df.sum(*columns)

    def mean(self, df: daft.DataFrame, *columns) -> daft.DataFrame:
        return df.mean(*columns)

    def count(self, df: daft.DataFrame) -> daft.DataFrame:
        return df.count()

    def min(self, df: daft.DataFrame, *columns) -> daft.DataFrame:
        return df.min(*columns)

    def max(self, df: daft.DataFrame, *columns) -> daft.DataFrame:
        return df.max(*columns)

    # Execution Methods
    def collect(self, df: daft.DataFrame) -> Any:
        return df.collect()

    def show(self, df: daft.DataFrame, n: int = 10) -> None:
        return df.show(n)

    def to_pandas(self, df: daft.DataFrame) -> Any:
        return df.to_pandas()

    def to_arrow(self, df: daft.DataFrame) -> Any:
        return df.to_arrow()

    def to_pydict(self, df: daft.DataFrame) -> Dict[str, Any]:
        return df.to_pydict()

    # Writing Methods
    def write_parquet(self, df: daft.DataFrame, path: str, **kwargs) -> daft.DataFrame:
        return df.write_parquet(path, **kwargs)

    def write_csv(self, df: daft.DataFrame, path: str, **kwargs) -> daft.DataFrame:
        return df.write_csv(path, **kwargs)

    # Schema and Lineage Methods
    def explain(self, df: daft.DataFrame) -> str:
        return df.explain()

    def schema(self, df: daft.DataFrame) -> Any:
        return df.schema

    def column_names(self, df: daft.DataFrame) -> List[str]:
        return df.column_names
