from __future__ import annotations

import os
import logging
from collections import deque
import tempo
import pyspark.sql.functions as f
from pyspark.sql import SparkSession
from pyspark.sql.utils import ParseException
from tsdf import TSDF
from typing import Optional

logger = logging.getLogger(__name__)


def readTimescaleDb(spark: SparkSession,
                    database_host_url: str,
                    port: int,
                    username: str,
                    password: str,
                    database_name: str,
                    table_name: str,
                    timeseries_col: Optional[str],
                    partition_col: Optional[list[str]],
                    num_partitions: Optional[int]):
    """
    Reads TimeScaleDB TimeSeries tables into TSDF.

    :param spark: SparkSession
    :param database_host_url: TimescaleDB Host
    :param port: TimescaleDB Port
    :param username: TimescaleDB username
    :param password: TimescaleDB password
    :param database_name: TimescaleDB database name
    :param table_name: TimescaleDB table name
    :param timeseries_col: Tempo TimeSeries Dataframe time column
    :param partition_col: Tempo Timeseries Dataframe time series identifier column(s)
    :param num_partitions: Number partitions to use for parallel reading from TimescaleDB
    :return: Tempo TSDF of Timescale Data
    """
    if num_partitions == None:
        get_partitions = (spark.read
                          .format("postgresql")
                          .option("query", f"SELECT num_chunks FROM timescaledb_information.hypertables "
                                           f"WHERE hypertable_name = '{table_name}';")
                          .option("host", database_host_url)
                          .option("port", port)
                          .option("user", username)
                          .option("password", password)
                          .load()
                          ).collect()[0][0]
        num_partitions = get_partitions

    get_partition_col = (spark.read
                         .format("postgresql")
                         .option("query", f"SELECT primary_dimension FROM timescaledb_information.chunks "
                                          f"WHERE hypertable_name = '{table_name}';")
                         .option("host", database_host_url)
                         .option("port", port)
                         .option("user", username)
                         .option("password", password)
                         .load()
                         ).collect()[0][0]
    bounds = (spark.read
              .format("postgresql")
              .option("query", f"SELECT max({get_partition_col}), min({get_partition_col}) FROM "
                               f"{database_name}.{table_name};")
              .option("host", database_host_url)
              .option("port", port)
              .option("user", username)
              .option("password", password)
              .option("numPartitions", num_partitions)
              .option("partitionColumn", get_partition_col)
              .load()
              ).collect()[0]
    upper_bound, lower_bound = bounds[0], bounds[1]
    raw_timescale_data = (spark.read
                          .format("postgresql")
                          .option("dbtable", table_name)
                          .option("host", database_host_url)
                          .option("port", port)
                          .option("database", database_name)
                          .option("user", username)
                          .option("password", password)
                          .option("numPartitions", num_partitions)
                          .option("lowerBound", lower_bound)
                          .option("upperBound", upper_bound)
                          .option("partitionColumn", get_partition_col)
                          .load()
                          )
    timescale_dimensions = (
        spark.read.format("postgresql")
        .option("query", f"SELECT * FROM timescaledb_information.dimensions "
                         f"WHERE hypertable_name = '{table_name}';")
        .option("host", database_host_url)
        .option("port", port)
        .option("user", username)
        .option("password", password)
        .load()
    )
    if timeseries_col is None:
        timeseries_col = timescale_dimensions.filter("dimension_type == 'Time'").select("column_name").rdd.flatMap(
            lambda x: x).collect()[0]
    if partition_col is None or partition_col is []:
        partition_col = timescale_dimensions.filter("dimension_type == 'Space'").select("column_name").rdd.flatMap(
            lambda x: x).collect()

    timescale_tsdf = TSDF(df=raw_timescale_data, ts_col=timeseries_col, partition_cols=partition_col)

    return timescale_tsdf


def writeTimescaleDb(spark: SparkSession,
                     tsdf: TSDF,
                     database_host_url: str,
                     port: int,
                     username: str,
                     password: str,
                     database_name: str,
                     table_name: str,
                     ):
    existence = (spark.read
                     .format("postgresql")
                     .option("query",
                             f"SELECT EXISTS(SELECT * FROM timescaledb_information.hypertables WHERE "
                             f"hypertable_name = '{table_name}');")
                     .option("host", database_host_url)
                     .option("port", port)
                     .option("user", username)
                     .option("password", password)
                     .load()
                     )
    existence_check = existence.count()
    if existence_check == 1:
        (tsdf.df
         .write
         .mode("append")
         .format("postgresql")
         .option("host", database_host_url)
         .option("port", port)
         .option("user", username)
         .option("password", password)
         .option("dbtable", f'{database_name}.{table_name}')
         .save())
    else:
        schema = tsdf.df.schema
        sql_schema = ", ".join([f"{field.name} {field.dataType.simpleString()}" for field in schema])
        (
            spark.read.format("postgresql")
            .option("query", f"CREATE TABLE {database_name}.{table_name} ({sql_schema});")
            .option("host", database_host_url)
            .option("port", port)
            .option("user", username)
            .option("password", password)
            .load()
        )
        space_partitions = ", ".join(str(element) for element in tsdf.partitionCols)
        (
            spark.read.format("postgresql")
            .option("query", f"SELECT create_distributed_hypertable({database_name}.{table_name}, {tsdf.ts_col}, {space_partitions}, replication_factor => 2)")
            .option("host", database_host_url)
            .option("port", port)
            .option("user", username)
            .option("password", password)
            .load()
        )
        (tsdf.df
         .write
         .mode("append")
         .format("postgresql")
         .option("host", database_host_url)
         .option("port", port)
         .option("user", username)
         .option("password", password)
         .option("dbtable", f'{database_name}.{table_name}')
         .save())



def write(
        tsdf: tempo.TSDF,
        spark: SparkSession,
        tabName: str,
        optimizationCols: list[str] = None,
):
    """
    param: tsdf: input TSDF object to write
    param: tabName Delta output table name
    param: optimizationCols list of columns to optimize on (time)
    """
    # hilbert curves more evenly distribute performance for querying multiple columns for Delta tables
    spark.conf.set("spark.databricks.io.skipping.mdc.curve", "hilbert")

    df = tsdf.df
    ts_col = tsdf.ts_col
    partitionCols = tsdf.partitionCols
    if optimizationCols:
        optimizationCols = optimizationCols + ["event_time"]
    else:
        optimizationCols = ["event_time"]

    useDeltaOpt = os.getenv("DATABRICKS_RUNTIME_VERSION") is not None

    view_df = df.withColumn("event_dt", f.to_date(f.col(ts_col))).withColumn(
        "event_time",
        f.translate(f.split(f.col(ts_col).cast("string"), " ")[1], ":", "").cast(
            "double"
        ),
    )
    view_cols = deque(view_df.columns)
    view_cols.rotate(1)
    view_df = view_df.select(*list(view_cols))

    view_df.write.mode("overwrite").partitionBy("event_dt").format("delta").saveAsTable(
        tabName
    )

    if useDeltaOpt:
        try:
            spark.sql(
                "optimize {} zorder by {}".format(
                    tabName, "(" + ",".join(partitionCols + optimizationCols) + ")"
                )
            )
        except ParseException as e:
            logger.error(
                "Delta optimizations attempted, but was not successful.\nError: {}".format(
                    e
                )
            )
    else:
        logger.warning(
            "Delta optimizations attempted on a non-Databricks platform. "
            "Switch to use Databricks Runtime to get optimization advantages."
        )
