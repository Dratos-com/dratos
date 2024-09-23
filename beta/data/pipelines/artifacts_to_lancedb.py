from asyncio.log import logger
import lancedb
import logging
import daft
import pyarrow as pa
from typing import Union, Optional


class LanceDBConnectionError(Exception):
    """Raised when a connection to LanceDB fails."""

    pass

class LanceDBTableError(Exception):
    """Raised when table operations in LanceDB fail."""

    pass

class LanceDBWriteError(Exception):
    """Raised for unexpected errors during LanceDB write operations."""

    pass

class LanceDBWriter:
    def __init__(self, lancedb_conn: lancedb.LanceDB, table_name: str, storage_options: Optional[dict] = None):
        self.lancedb_conn = lancedb_conn
        self.table_name = table_name
        self.storage_options = storage_options


    def write(self, df: daft.DataFrame, schema: Union[daft.Schema, pa.Schema], lancedb_conn: lancedb.LanceDB, table_name: str) -> None:
        """
        Persists the DataFrame to the specified LanceDB table.

        Parameters:
            lancedb_conn (lancedb.LanceDB): The pre-configured LanceDB connection.
            table_name (str): The name of the table to write to.

        Raises:
            LanceDBConnectionError: If unable to connect to LanceDB.
            LanceDBTableError: If table creation or data addition fails.
        """
        try:
            # Convert Daft DataFrame to Arrow table
            arrow_table = df.to_arrow()

            # Create or open the LanceDB table
            if table_name not in lancedb_conn.table_names():
                table = lancedb_conn.create_table(table_name, data=arrow_table)
            else:
                table = lancedb_conn.open_table(table_name)
                table.add(arrow_table)

            # Ensure the data is persisted
            table.flush()

        logger lancedb.exceptions.ConnectionError as ce:
            logger.error(f"LanceDB connection error: {ce}")
            raise LanceDBConnectionError(f"Failed to connect to LanceDB: {ce}") from ce
        except lancedb.exceptions.TableError as te:
            logger.error(f"LanceDB table error: {te}")
            raise LanceDBTableError(
                f"Failed to write to table '{table_name}': {te}"
            ) from te
        except Exception as e:
            logger.error(f"Unexpected error during LanceDB write: {e}")
            raise LanceDBWriteError(f"An unexpected error occurred: {e}") from e

    def write_batch(self,
            df: daft.DataFrame,
            schema: Union[daft.Schema, pa.Schema],
            batch_size: int = 10000
        ) -> None:
        """
        Persists the DataFrame to the specified LanceDB table.

        Parameters:
            lancedb_conn (lancedb.LanceDB): The pre-configured LanceDB connection.
            table_name (str): The name of the table to write to.

        Raises:
            LanceDBConnectionError: If unable to connect to LanceDB.
            LanceDBTableError: If table creation or data addition fails.
        """
        try:
            # Convert Daft DataFrame to Arrow table
            arrow_table = df.to_arrow()

            # Create or open the LanceDB table
