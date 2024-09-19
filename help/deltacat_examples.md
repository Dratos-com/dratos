from deltacat.storage import interface as unimplemented_deltacat_storage


def run_all(dc_storage=unimplemented_deltacat_storage):
    """Run all examples."""

    """
    Example list_namespaces() result containing a single namespace:
    {
        'items': [
            {
                'namespace': 'TestNamespace',
            }
        ],
        'pagination_key': 'dmVyc2lvbmVkVGFibGVOYW1l'
    }
    """
    namespaces = []
    namespaces_list_result = dc_storage.list_namespaces()
    while namespaces_list_result:
        namespaces_list_result = namespaces_list_result.next_page()
        namespaces.extend(namespaces_list_result.read_page())

    print(f"All Namespaces: {namespaces}")

    """
    Example list_tables() result containing a single table:
    {
        'items': [
            {
                'id': {
                    'namespace': 'TestNamespace',
                    'tableName': 'TestTable'
                },
                'description': 'Test table description.',
                'properties': {
                    'testPropertyOne': 'testValueOne',
                    'testPropertyTwo': 'testValueTwo'
                }
            }
        ],
       'pagination_key': 'dmVyc2lvbmVkVGFibGVOYW1l'
    }
    """
    test_tables = []
    tables_list_result = dc_storage.list_tables("TestNamespace")
    while tables_list_result:
        tables_list_result = tables_list_result.next_page()
        test_tables.extend(tables_list_result.read_page())

    print(f"All 'TestNamespace' Tables: {test_tables}")

    """
    Example list_partitions() result containing a single partition:
    {
        'items': [
            {
                'partitionKeyValues': ['1', '2017-08-31T00:00:00.000Z']
            }
        ],
        'pagination_key': 'dmVyc2lvbmVkVGFibGVOYW1l'
    }
    """
    # Partitions will automatically be returned for the latest active version of
    # the specified table.
    table_partitions = []
    partitions_list_result = dc_storage.list_partitions(
        "TestNamespace",
        "TestTable",
    )
    while partitions_list_result:
        partitions_list_result = partitions_list_result.next_page()
        table_partitions.extend(partitions_list_result.read_page())
    print(f"All Table Partitions: {table_partitions}")

    """
    Example list_deltas result containing a single delta:
    {
        'items': [
            {
                'type': 'upsert',
                'locator": {
                    'streamPosition': 1551898425276,
                    'partitionLocator': {
                        'partitionId': 'de75623a-7adf-4cf0-b982-7b514502be82'
                        'partitionValues': ['1', '2018-03-06T00:00:00.000Z'],
                        'streamLocator': {
                            'namespace': 'TestNamespace',
                            'tableName': 'TestTable',
                            'tableVersion': '1',
                            'streamId': 'dbcbbf56-4bcb-4b94-8cf2-1c6d57ccfe74',
                            'storageType': 'AwsGlueCatalog'
                        }
                    }
                },
                'properties': {
                    'parent_stream_position': '1551898423165'
                },
                'meta': {
                    'contentLength': 9423157342,
                    'fileCount': 117,
                    'recordCount': 188463146,
                    'sourceContentLength': 37692629368,
                }
            }
        ],
        'paginationKey': 'enQzd3mqcnNkQIFkaHQ1ZW2m'
    }
    """
    # Deltas will automatically be returned for the latest active version of the
    # specified table.
    deltas_list_result = dc_storage.list_deltas(
        "TestNamespace",
        "TestTable",
        ["1", "2018-03-06T00:00:00.000Z"],
    )
    all_partition_deltas = deltas_list_result.all_items()
    print(f"All Partition Deltas: {all_partition_deltas}")


if __name__ == "__main__":
    run_all()



import ray

from deltacat.storage import interface as unimplemented_deltacat_storage

ray.init(address="auto")


def run_all(dc_storage_ray=unimplemented_deltacat_storage):
    """Run all examples."""

    # make an asynchronous call to list namespaces
    list_namespaces_future = dc_storage_ray.list_namespaces.remote()

    # gather the first page of namespaces synchronously
    namespaces_page_one = ray.get(list_namespaces_future)
    print(f"First page of Namespaces: {namespaces_page_one}")

    # make asynchronous invocations to list tables for the first 10 namespaces
    pending_futures = []
    for i in range(10):
        namespace = namespaces_page_one.read_page()[i]["namespace"]
        list_tables_future = dc_storage_ray.list_tables.remote(namespace)
        pending_futures.append(list_tables_future)

    # asynchronously gather each table listing in the order they complete
    while len(pending_futures):
        ready_futures, pending_futures = ray.wait(pending_futures)
        tables = ray.get(ready_futures[0])
        print(f"Received one page of tables: {tables}")


if __name__ == "__main__":
    run_all()


import ray
from typing import List
import pyarrow as pa
import daft
from typing import Dict, Any, List
import lancedb
from lancedb.pydantic import 
from deltacat.storage import interface as unimplemented_deltacat_storage
from deltacat.types.media import StorageType
from deltacat.utils.performance import timed_invocation

ray.init(address="auto")

@ray.remote
def convert_sort_and_dedupe(pyarrow_table):
    pandas_dataframe = pyarrow_table.to_pandas()
    pandas_dataframe.sort_values(["sort_key_1"])
    pandas_dataframe.drop_duplicates(["dedupe_key_1", "dedupe_key_2"])
    return pandas_dataframe


def run_all(dc_storage_ray=unimplemented_deltacat_storage):
    """Run all examples."""
    deltas_list_result = ray.get(
        dc_storage_ray.list_deltas.remote(
            "TestProvider",
            "TestTable",
            ["1", "2018-03-06T00:00:00.000Z"],
        )
    )

    delta = deltas_list_result.read_page()[0]

    pa_table_pending_ids = ray.get(
        dc_storage_ray.download_delta.remote(delta),
        storage_type=StorageType.LOCAL,
    )

    pending_futures = []
    for table_pending_id in pa_table_pending_ids:
        pending_future = convert_sort_and_dedupe.remote(table_pending_id)
        pending_futures.append(pending_future)
    pandas_dataframes, latency = timed_invocation(ray.get, pending_futures)
    print(f"Time to read, convert, sort, and dedupe delta: {latency}s")


if __name__ == "__main__":
    run_all()


import ray

from deltacat import ListResult
from deltacat.storage import interface as unimplemented_deltacat_storage
from deltacat.utils.ray_utils.collections import DistributedCounter
from deltacat.utils.ray_utils.performance import invoke_with_perf_counter

ray.init(address="auto")


def list_all_tables_for_namespaces(
    namespaces, dc_storage=unimplemented_deltacat_storage
):

    namespace_tables_promises = {}
    for namespace in namespaces:
        namespace = namespace["namespace"]
        tables_list_result_promise = ListResult.all_items_ray.remote(
            dc_storage.list_tables(namespace)
        )
        namespace_tables_promises[namespace] = tables_list_result_promise

    namespace_table_counts = {}
    tables = []
    for namespace, promise in namespace_tables_promises.items():
        namespace_tables = ray.get(promise)
        namespace_table_count = len(namespace_tables)
        namespace_table_counts[namespace] = namespace_table_count
        tables.extend(namespace_tables)
    sorted_namespace_table_counts = dict(
        sorted(namespace_table_counts.items(), key=lambda item: item[1])
    )
    print(f"Table counts by namespace: {sorted_namespace_table_counts}")
    print(f"Total tables: {len(tables)}")

    return tables


def run_all(dc_storage=unimplemented_deltacat_storage):
    """Run all examples."""

    distributed_counter = DistributedCounter.remote()
    namespaces, latency = invoke_with_perf_counter(
        distributed_counter,
        "list_all_namespaces",
        dc_storage.list_namespaces().all_items,
    )
    print(f"Total namespaces: {len(namespaces)}")
    print(f"List namespace latency: {latency}")

    tables, latency = invoke_with_perf_counter(
        distributed_counter,
        "list_all_tables",
        list_all_tables_for_namespaces,
        namespaces,
        dc_storage,
    )
    print(f"List tables latency: {latency}")


if __name__ == "__main__":
    run_all()


