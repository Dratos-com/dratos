DataFrame
DataFrame

A Daft DataFrame is a table of data.

Note

Most DataFrame methods are lazy, meaning that they do not execute computation immediately when invoked. Instead, these operations are enqueued in the DataFrame’s internal query plan, and are only executed when Execution DataFrame methods are called.

Data Manipulation
Selecting Columns
DataFrame.__getitem__

Gets a column from the DataFrame as an Expression (df["mycol"])

Manipulating Columns
DataFrame.select

Creates a new DataFrame from the provided expressions, similar to a SQL SELECT

DataFrame.with_column

Adds a column to the current DataFrame with an Expression, equivalent to a select with all current columns and the new one

DataFrame.with_columns

Adds columns to the current DataFrame with Expressions, equivalent to a select with all current columns and the new ones

DataFrame.pivot

Pivots a column of the DataFrame and performs an aggregation on the values.

DataFrame.exclude

Drops columns from the current DataFrame by name

DataFrame.explode

Explodes a List column, where every element in each row's List becomes its own row, and all other columns in the DataFrame are duplicated across rows

DataFrame.unpivot

Unpivots a DataFrame from wide to long format.

DataFrame.melt

Alias for unpivot

DataFrame.transform

Apply a function that takes and returns a DataFrame.

Filtering Rows
DataFrame.distinct

Computes unique rows, dropping duplicates

DataFrame.where

Filters rows via a predicate expression, similar to SQL WHERE.

DataFrame.limit

Limits the rows in the DataFrame to the first N rows, similar to a SQL LIMIT

DataFrame.sample

Samples a fraction of rows from the DataFrame

Reordering
DataFrame.sort

Sorts DataFrame globally

DataFrame.repartition

Repartitions DataFrame to num partitions

DataFrame.into_partitions

Splits or coalesces DataFrame to num partitions.

Combining
DataFrame.join

Column-wise join of the current DataFrame with an other DataFrame, similar to a SQL JOIN

DataFrame.concat

Concatenates two DataFrames together in a "vertical" concatenation.

Aggregations
DataFrame.groupby

Performs a GroupBy on the DataFrame for aggregation

DataFrame.sum

Performs a global sum on the DataFrame

DataFrame.mean

Performs a global mean on the DataFrame

DataFrame.count

Performs a global count on the DataFrame

DataFrame.min

Performs a global min on the DataFrame

DataFrame.max

Performs a global max on the DataFrame

DataFrame.agg

Perform aggregations on this DataFrame.

Execution
Note

These methods will execute the operations in your DataFrame and are blocking.

Data Retrieval
These methods will run the dataframe and retrieve them to where the code is being run.

DataFrame.to_pydict

Converts the current DataFrame to a python dictionary.

DataFrame.iter_partitions

Begin executing this dataframe and return an iterator over the partitions.

DataFrame.iter_rows

Return an iterator of rows for this dataframe.

Materialization
DataFrame.collect

Executes the entire DataFrame and materializes the results

Visualization
DataFrame.show

Executes enough of the DataFrame in order to display the first n rows

Writing Data
DataFrame.write_parquet

Writes the DataFrame as parquet files, returning a new DataFrame with paths to the files that were written

DataFrame.write_csv

Writes the DataFrame as CSV files, returning a new DataFrame with paths to the files that were written

DataFrame.write_iceberg

Writes the DataFrame to an Iceberg table, returning a new DataFrame with the operations that occurred.

DataFrame.write_deltalake

Writes the DataFrame to a Delta Lake table, returning a new DataFrame with the operations that occurred.

Integrations
DataFrame.to_arrow

Converts the current DataFrame to a pyarrow Table.

DataFrame.to_pandas

Converts the current DataFrame to a pandas DataFrame.

DataFrame.to_torch_map_dataset

Convert the current DataFrame into a map-style Torch Dataset for use with PyTorch.

DataFrame.to_torch_iter_dataset

Convert the current DataFrame into a Torch IterableDataset for use with PyTorch.

DataFrame.to_ray_dataset

Converts the current DataFrame to a Ray Dataset which is useful for running distributed ML model training in Ray

DataFrame.to_dask_dataframe

Converts the current Daft DataFrame to a Dask DataFrame.

Schema and Lineage
DataFrame.explain

Prints the (logical and physical) plans that will be executed to produce this DataFrame.

DataFrame.schema

Returns the Schema of the DataFrame, which provides information about each column

DataFrame.column_names

Returns column names of DataFrame as a list of strings.