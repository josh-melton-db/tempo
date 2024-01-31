from parameterized import parameterized

from tempo.tsdf import TSDF
from tempo.tsschema import (
    SimpleTimestampIndex,
    SimpleDateIndex,
    OrdinalTSIndex,
    ParsedTimestampIndex,
    ParsedDateIndex,
    TSSchema,
)
from tests.base import TestDataFrame, SparkTest
import pyspark.sql.functions as sfn
from pyspark.sql.types import (
    StructField,
    StructType,
    StringType,
    TimestampType,
    DateType,
    DoubleType,
    IntegerType,
)


class TSDFBaseTests(SparkTest):
    @parameterized.expand(
        [
            ("simple_ts_idx", SimpleTimestampIndex),
            ("simple_ts_no_series", SimpleTimestampIndex),
            ("simple_date_idx", SimpleDateIndex),
            ("ordinal_double_index", OrdinalTSIndex),
            ("ordinal_int_index", OrdinalTSIndex),
            ("parsed_ts_idx", ParsedTimestampIndex),
            ("parsed_date_idx", ParsedDateIndex),
        ]
    )
    def test_tsdf_constructor(self, init_tsdf_id, expected_idx_class):
        # create TSDF
        init_tsdf = self.get_test_data(init_tsdf_id).as_tsdf()
        # check that TSDF was created correctly
        self.assertIsNotNone(init_tsdf)
        self.assertIsInstance(init_tsdf, TSDF)
        # validate the TSSchema
        self.assertIsNotNone(init_tsdf.ts_schema)
        self.assertIsInstance(init_tsdf.ts_schema, TSSchema)
        # validate the TSIndex
        self.assertIsNotNone(init_tsdf.ts_index)
        self.assertIsInstance(init_tsdf.ts_index, expected_idx_class)

    @parameterized.expand(
        [
            ("simple_ts_idx", ["symbol"]),
            ("simple_ts_no_series", []),
            ("simple_date_idx", ["station"]),
            ("ordinal_double_index", ["symbol"]),
            ("ordinal_int_index", ["symbol"]),
            ("parsed_ts_idx", ["symbol"]),
            ("parsed_date_idx", ["station"]),
        ]
    )
    def test_series_ids(self, init_tsdf_id, expected_series_ids):
        # load TSDF
        tsdf = self.get_test_data(init_tsdf_id).as_tsdf()
        # validate series ids
        self.assertEqual(set(tsdf.series_ids), set(expected_series_ids))

    @parameterized.expand(
        [
            ("simple_ts_idx", ["event_ts", "symbol"]),
            ("simple_ts_no_series", ["event_ts"]),
            ("simple_date_idx", ["date", "station"]),
            ("ordinal_double_index", ["event_ts_dbl", "symbol"]),
            ("ordinal_int_index", ["order", "symbol"]),
            ("parsed_ts_idx", ["ts_idx", "symbol"]),
            ("parsed_date_idx", ["ts_idx", "station"]),
        ]
    )
    def test_structural_cols(self, init_tsdf_id, expected_structural_cols):
        # load TSDF
        tsdf = self.get_test_data(init_tsdf_id).as_tsdf()
        # validate structural cols
        self.assertEqual(set(tsdf.structural_cols), set(expected_structural_cols))

    @parameterized.expand(
        [
            ("simple_ts_idx", ["trade_pr"]),
            ("simple_ts_no_series", ["trade_pr"]),
            ("simple_date_idx", ["temp"]),
            ("ordinal_double_index", ["trade_pr"]),
            ("ordinal_int_index", ["trade_pr"]),
            ("parsed_ts_idx", ["trade_pr"]),
            ("parsed_date_idx", ["temp"]),
        ]
    )
    def test_obs_cols(self, init_tsdf_id, expected_obs_cols):
        # load TSDF
        tsdf = self.get_test_data(init_tsdf_id).as_tsdf()
        # validate obs cols
        self.assertEqual(set(tsdf.observational_cols), set(expected_obs_cols))

    @parameterized.expand(
        [
            ("simple_ts_idx", ["trade_pr"]),
            ("simple_ts_no_series", ["trade_pr"]),
            ("simple_date_idx", ["temp"]),
            ("ordinal_double_index", ["trade_pr"]),
            ("ordinal_int_index", ["trade_pr"]),
            ("parsed_ts_idx", ["trade_pr"]),
            ("parsed_date_idx", ["temp"]),
        ]
    )
    def test_metric_cols(self, init_tsdf_id, expected_metric_cols):
        # load TSDF
        tsdf = self.get_test_data(init_tsdf_id).as_tsdf()
        # validate metric cols
        self.assertEqual(set(tsdf.metric_cols), set(expected_metric_cols))

    @parameterized.expand(
        [
            ("simple_ts_idx", ["event_ts", "symbol"], {"event_ts", "symbol"}),
            ("simple_ts_no_series", ["event_ts", "trade_pr"], {"event_ts", "trade_pr"}),
            (
                "simple_date_idx",
                ["date", "station", "temp"],
                {"date", "station", "temp"},
            ),
            ("ordinal_double_index", ["*"], {"event_ts_dbl", "symbol", "trade_pr"}),
            (
                "ordinal_int_index",
                ["order", "symbol", "trade_pr"],
                {"order", "symbol", "trade_pr"},
            ),
            (
                "parsed_ts_idx",
                ["ts_idx", "trade_pr", "symbol"],
                {"ts_idx", "trade_pr", "symbol"},
            ),
            ("parsed_date_idx", "*", {"ts_idx", "station", "temp"}),
        ]
    )
    def test_select(self, init_tsdf_id, col_to_select, expected_tsdf_cols):
        # load TSDF
        tsdf = self.get_test_data(init_tsdf_id).as_tsdf()
        # select column
        selected_tsdf = tsdf.select(col_to_select)
        # validate selected column
        self.assertEqual(set(selected_tsdf.df.columns), expected_tsdf_cols)

    @parameterized.expand(
        [
            ("simple_ts_idx", ["event_ts"]),
            ("simple_ts_no_series", ["event_ts", "trade_pr"]),
            ("simple_date_idx", ["station", "temp"]),
            ("ordinal_double_index", ["symbol", "trade_pr"]),
            ("ordinal_int_index", ["order", "trade_pr"]),
            ("parsed_ts_idx", ["ts_idx", "symbol"]),
            ("parsed_date_idx", ["station"]),
        ]
    )
    def test_select_without_structural(self, init_tsdf_id, col_to_select):
        # load TSDF
        tsdf = self.get_test_data(init_tsdf_id).as_tsdf()
        # validate selecting without structural column throws an error
        self.assertRaises(AssertionError, tsdf.select)

    @parameterized.expand(
        [
            (
                "simple_ts_idx",
                "symbol = 'S1'",
                {
                    "ts_idx": {"ts_col": "event_ts", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, event_ts string, trade_pr float",
                        "ts_convert": ["event_ts"],
                        "data": [
                            ["S1", "2020-08-01 00:00:10", 349.21],
                            ["S1", "2020-08-01 00:01:12", 351.32],
                            ["S1", "2020-09-01 00:02:10", 361.1],
                            ["S1", "2020-09-01 00:19:12", 362.1],
                        ],
                    },
                },
            ),
            (
                "simple_ts_no_series",
                "event_ts > '2020-08-01 00:01:20'",
                {
                    "ts_idx": {"ts_col": "event_ts", "series_ids": []},
                    "df": {
                        "schema": "event_ts string, trade_pr float",
                        "ts_convert": ["event_ts"],
                        "data": [
                            ["2020-08-01 00:01:24", 751.92],
                            ["2020-09-01 00:02:10", 361.1],
                            ["2020-09-01 00:19:12", 362.1],
                            ["2020-09-01 00:20:42", 762.33],
                        ],
                    },
                },
            ),
            (
                "simple_date_idx",
                "date = '2020-08-02'",
                {
                    "ts_idx": {"ts_col": "date", "series_ids": ["station"]},
                    "df": {
                        "schema": "station string, date string, temp float",
                        "date_convert": ["date"],
                        "data": [
                            ["LGA", "2020-08-02", 28.79],
                            ["YYZ", "2020-08-02", 22.25],
                        ],
                    },
                },
            ),
            (
                "ordinal_double_index",
                "trade_pr < 500",
                {
                    "ts_idx": {"ts_col": "event_ts_dbl", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, event_ts_dbl double, trade_pr float",
                        "data": [
                            ["S1", 0.13, 349.21],
                            ["S1", 1.207, 351.32],
                            ["S1", 10.0, 361.1],
                            ["S1", 24.357, 362.1],
                        ],
                    },
                },
            ),
            (
                "ordinal_int_index",
                "order > 10 and trade_pr < 750",
                {
                    "ts_idx": {"ts_col": "order", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, order int, trade_pr float",
                        "data": [
                            ["S1", 20, 351.32],
                            ["S1", 127, 361.1],
                            ["S1", 243, 362.1],
                        ],
                    },
                },
            ),
            (
                "parsed_ts_idx",
                "symbol='S2' or event_ts >= '2020-09-01 00:02:10.032'",
                {
                    "ts_idx": {
                        "ts_col": "event_ts",
                        "series_ids": ["symbol"],
                        "ts_fmt": "yyyy-MM-dd HH:mm:ss.SSS",
                    },
                    "tsdf_constructor": "fromStringTimestamp",
                    "df": {
                        "schema": "symbol string, event_ts string, trade_pr float",
                        "data": [
                            ["S1", "2020-09-01 00:02:10.032", 361.1],
                            ["S1", "2020-09-01 00:19:12.043", 362.1],
                            ["S2", "2020-08-01 00:01:10.054", 743.01],
                            ["S2", "2020-08-01 00:01:24.065", 751.92],
                            ["S2", "2020-09-01 00:02:10.076", 761.10],
                            ["S2", "2020-09-01 00:20:42.087", 762.33],
                        ],
                    },
                },
            ),
            (
                "parsed_date_idx",
                "station='LGA' and (date >= '2020-08-03' or temp < 28)",
                {
                    "ts_idx": {
                        "ts_col": "date",
                        "series_ids": ["station"],
                        "ts_fmt": "yyyy-MM-dd",
                    },
                    "tsdf_constructor": "fromStringTimestamp",
                    "df": {
                        "schema": "station string, date string, temp float",
                        "data": [
                            ["LGA", "2020-08-01", 27.58],
                            ["LGA", "2020-08-03", 28.53],
                            ["LGA", "2020-08-04", 25.57],
                        ],
                    },
                },
            ),
        ]
    )
    def test_where(self, init_tsdf_id, condition, expected_tsdf_dict):
        # load TSDF
        tsdf = self.get_test_data(init_tsdf_id).as_tsdf()
        # filter on condition
        where_tsdf = tsdf.where(condition)
        # validate the filter
        expected_tsdf = TestDataFrame(self.spark, expected_tsdf_dict).as_tsdf()
        self.assertDataFrameEquality(where_tsdf, expected_tsdf)


class BaseTransformationTests(SparkTest):
    @parameterized.expand(
        [
            (
                "simple_ts_idx",
                {
                    "ts_idx": {"ts_col": "event_ts", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, event_ts string, trade_pr float",
                        "ts_convert": ["event_ts"],
                        "data": [
                            ["S1", "2020-08-01 00:00:10", 349.21],
                            ["S2", "2020-09-01 00:20:42", 762.33],
                            ["S1", "2020-09-01 00:19:12", 362.1],
                            ["S2", "2020-08-01 00:01:10", 743.01],
                            ["S2", "2020-08-01 00:01:24", 751.92],
                            ["S1", "2020-08-01 00:01:12", 351.32],
                            ["S1", "2020-09-01 00:02:10", 361.1],
                            ["S2", "2020-09-01 00:02:10", 761.10],
                        ],
                    },
                },
            ),
            (
                "simple_ts_no_series",
                {
                    "ts_idx": {"ts_col": "event_ts", "series_ids": []},
                    "df": {
                        "schema": "event_ts string, trade_pr float",
                        "ts_convert": ["event_ts"],
                        "data": [
                            ["2020-09-01 00:02:10", 361.1],
                            ["2020-08-01 00:00:10", 349.21],
                            ["2020-08-01 00:01:12", 351.32],
                            ["2020-09-01 00:19:12", 362.1],
                            ["2020-08-01 00:01:24", 751.92],
                            ["2020-09-01 00:20:42", 762.33],
                            ["2020-08-01 00:01:10", 743.01],
                        ],
                    },
                },
            ),
            (
                "simple_date_idx",
                {
                    "ts_idx": {"ts_col": "date", "series_ids": ["station"]},
                    "df": {
                        "schema": "station string, date string, temp float",
                        "date_convert": ["date"],
                        "data": [
                            ["YYZ", "2020-08-04", 20.65],
                            ["LGA", "2020-08-04", 25.57],
                            ["LGA", "2020-08-02", 28.79],
                            ["LGA", "2020-08-03", 28.53],
                            ["LGA", "2020-08-01", 27.58],
                            ["YYZ", "2020-08-02", 22.25],
                            ["YYZ", "2020-08-01", 24.16],
                            ["YYZ", "2020-08-03", 20.62],
                        ],
                    },
                },
            ),
            (
                "ordinal_double_index",
                {
                    "ts_idx": {"ts_col": "event_ts_dbl", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, event_ts_dbl double, trade_pr float",
                        "data": [
                            ["S2", 0.005, 743.01],
                            ["S2", 0.1, 751.92],
                            ["S2", 1.0, 761.10],
                            ["S1", 1.207, 351.32],
                            ["S1", 0.13, 349.21],
                            ["S1", 24.357, 362.1],
                            ["S2", 10.0, 762.33],
                            ["S1", 10.0, 361.1],
                        ],
                    },
                },
            ),
            (
                "ordinal_int_index",
                {
                    "ts_idx": {"ts_col": "order", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, order int, trade_pr float",
                        "data": [
                            ["S2", 0, 743.01],
                            ["S2", 100, 762.33],
                            ["S1", 1, 349.21],
                            ["S1", 127, 361.1],
                            ["S1", 243, 362.1],
                            ["S2", 1, 751.92],
                            ["S2", 10, 761.10],
                            ["S1", 20, 351.32],
                        ],
                    },
                },
            ),
            (
                "parsed_ts_idx",
                {
                    "ts_idx": {
                        "ts_col": "event_ts",
                        "series_ids": ["symbol"],
                        "ts_fmt": "yyyy-MM-dd HH:mm:ss.SSS",
                    },
                    "tsdf_constructor": "fromStringTimestamp",
                    "df": {
                        "schema": "symbol string, event_ts string, trade_pr float",
                        "data": [
                            ["S1", "2020-08-01 00:00:10.010", 349.21],
                            ["S1", "2020-09-01 00:02:10.032", 361.1],
                            ["S1", "2020-08-01 00:01:12.021", 351.32],
                            ["S2", "2020-08-01 00:01:10.054", 743.01],
                            ["S1", "2020-09-01 00:19:12.043", 362.1],
                            ["S2", "2020-08-01 00:01:24.065", 751.92],
                            ["S2", "2020-09-01 00:20:42.087", 762.33],
                            ["S2", "2020-09-01 00:02:10.076", 761.10],
                        ],
                    },
                },
            ),
            (
                "parsed_date_idx",
                {
                    "ts_idx": {
                        "ts_col": "date",
                        "series_ids": ["station"],
                        "ts_fmt": "yyyy-MM-dd",
                    },
                    "tsdf_constructor": "fromStringTimestamp",
                    "df": {
                        "schema": "station string, date string, temp float",
                        "data": [
                            ["YYZ", "2020-08-03", 20.62],
                            ["LGA", "2020-08-02", 28.79],
                            ["YYZ", "2020-08-04", 20.65],
                            ["YYZ", "2020-08-01", 24.16],
                            ["LGA", "2020-08-01", 27.58],
                            ["LGA", "2020-08-04", 25.57],
                            ["YYZ", "2020-08-02", 22.25],
                            ["LGA", "2020-08-03", 28.53],
                        ],
                    },
                },
            ),
        ]
    )
    def test_withNaturalOrdering(self, init_tsdf_id, scrambled_df_dict):
        # load init TSDF
        tsdf = self.get_test_data(init_tsdf_id).as_tsdf()
        # load scrambled TSDF
        scrambed_tsdf = TestDataFrame(self.spark, scrambled_df_dict).as_tsdf()
        # natural reorder scrambled TSDF
        ordered_tsdf = scrambed_tsdf.withNaturalOrdering()
        # validate the reorder
        self.assertDataFrameEquality(ordered_tsdf, tsdf)

    @parameterized.expand(
        [
            (
                "simple_ts_idx",
                {
                    "ts_idx": {"ts_col": "event_ts", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, event_ts string, trade_pr float, new_col int",
                        "ts_convert": ["event_ts"],
                        "data": [
                            ["S1", "2020-08-01 00:00:10", 349.21, 1],
                            ["S1", "2020-08-01 00:01:12", 351.32, 1],
                            ["S1", "2020-09-01 00:02:10", 361.1, 1],
                            ["S1", "2020-09-01 00:19:12", 362.1, 1],
                            ["S2", "2020-08-01 00:01:10", 743.01, 1],
                            ["S2", "2020-08-01 00:01:24", 751.92, 1],
                            ["S2", "2020-09-01 00:02:10", 761.10, 1],
                            ["S2", "2020-09-01 00:20:42", 762.33, 1],
                        ],
                    },
                },
            ),
            (
                "simple_ts_no_series",
                {
                    "ts_idx": {"ts_col": "event_ts", "series_ids": []},
                    "df": {
                        "schema": "event_ts string, trade_pr float, new_col int",
                        "ts_convert": ["event_ts"],
                        "data": [
                            ["2020-08-01 00:00:10", 349.21, 1],
                            ["2020-08-01 00:01:10", 743.01, 1],
                            ["2020-08-01 00:01:12", 351.32, 1],
                            ["2020-08-01 00:01:24", 751.92, 1],
                            ["2020-09-01 00:02:10", 361.1, 1],
                            ["2020-09-01 00:19:12", 362.1, 1],
                            ["2020-09-01 00:20:42", 762.33, 1],
                        ],
                    },
                },
            ),
            (
                "simple_date_idx",
                {
                    "ts_idx": {"ts_col": "date", "series_ids": ["station"]},
                    "df": {
                        "schema": "station string, date string, temp float, new_col int",
                        "date_convert": ["date"],
                        "data": [
                            ["LGA", "2020-08-01", 27.58, 1],
                            ["LGA", "2020-08-02", 28.79, 1],
                            ["LGA", "2020-08-03", 28.53, 1],
                            ["LGA", "2020-08-04", 25.57, 1],
                            ["YYZ", "2020-08-01", 24.16, 1],
                            ["YYZ", "2020-08-02", 22.25, 1],
                            ["YYZ", "2020-08-03", 20.62, 1],
                            ["YYZ", "2020-08-04", 20.65, 1],
                        ],
                    },
                },
            ),
            (
                "ordinal_double_index",
                {
                    "ts_idx": {"ts_col": "event_ts_dbl", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, event_ts_dbl double, trade_pr float, new_col int",
                        "data": [
                            ["S1", 0.13, 349.21, 1],
                            ["S1", 1.207, 351.32, 1],
                            ["S1", 10.0, 361.1, 1],
                            ["S1", 24.357, 362.1, 1],
                            ["S2", 0.005, 743.01, 1],
                            ["S2", 0.1, 751.92, 1],
                            ["S2", 1.0, 761.10, 1],
                            ["S2", 10.0, 762.33, 1],
                        ],
                    },
                },
            ),
            (
                "ordinal_int_index",
                {
                    "ts_idx": {"ts_col": "order", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, order int, trade_pr float, new_col int",
                        "data": [
                            ["S1", 1, 349.21, 1],
                            ["S1", 20, 351.32, 1],
                            ["S1", 127, 361.1, 1],
                            ["S1", 243, 362.1, 1],
                            ["S2", 0, 743.01, 1],
                            ["S2", 1, 751.92, 1],
                            ["S2", 10, 761.10, 1],
                            ["S2", 100, 762.33, 1],
                        ],
                    },
                },
            ),
            (
                "parsed_ts_idx",
                {
                    "ts_idx": {
                        "ts_col": "event_ts",
                        "series_ids": ["symbol"],
                        "ts_fmt": "yyyy-MM-dd HH:mm:ss.SSS",
                    },
                    "tsdf_constructor": "fromStringTimestamp",
                    "df": {
                        "schema": "symbol string, event_ts string, trade_pr float, new_col int",
                        "data": [
                            ["S1", "2020-08-01 00:00:10.010", 349.21, 1],
                            ["S1", "2020-08-01 00:01:12.021", 351.32, 1],
                            ["S1", "2020-09-01 00:02:10.032", 361.1, 1],
                            ["S1", "2020-09-01 00:19:12.043", 362.1, 1],
                            ["S2", "2020-08-01 00:01:10.054", 743.01, 1],
                            ["S2", "2020-08-01 00:01:24.065", 751.92, 1],
                            ["S2", "2020-09-01 00:02:10.076", 761.10, 1],
                            ["S2", "2020-09-01 00:20:42.087", 762.33, 1],
                        ],
                    },
                },
            ),
            (
                "parsed_date_idx",
                {
                    "ts_idx": {
                        "ts_col": "date",
                        "series_ids": ["station"],
                        "ts_fmt": "yyyy-MM-dd",
                    },
                    "tsdf_constructor": "fromStringTimestamp",
                    "df": {
                        "schema": "station string, date string, temp float, new_col int",
                        "data": [
                            ["LGA", "2020-08-01", 27.58, 1],
                            ["LGA", "2020-08-02", 28.79, 1],
                            ["LGA", "2020-08-03", 28.53, 1],
                            ["LGA", "2020-08-04", 25.57, 1],
                            ["YYZ", "2020-08-01", 24.16, 1],
                            ["YYZ", "2020-08-02", 22.25, 1],
                            ["YYZ", "2020-08-03", 20.62, 1],
                            ["YYZ", "2020-08-04", 20.65, 1],
                        ],
                    },
                },
            ),
        ]
    )
    def test_withColumn(self, init_tsdf_id, expected_tsdf_dict):
        # load TSDF
        tsdf = self.get_test_data(init_tsdf_id).as_tsdf()
        # add column
        newcol_tsdf = tsdf.withColumn("new_col", sfn.lit(1))
        # validate the added column
        expected_tsdf = TestDataFrame(self.spark, expected_tsdf_dict).as_tsdf()
        self.assertDataFrameEquality(newcol_tsdf, expected_tsdf)

    @parameterized.expand(
        [
            (
                "simple_ts_idx",
                {
                    "ts_idx": {"ts_col": "event_ts", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, event_ts string, new_col float",
                        "ts_convert": ["event_ts"],
                        "data": [
                            ["S1", "2020-08-01 00:00:10", 349.21],
                            ["S1", "2020-08-01 00:01:12", 351.32],
                            ["S1", "2020-09-01 00:02:10", 361.1],
                            ["S1", "2020-09-01 00:19:12", 362.1],
                            ["S2", "2020-08-01 00:01:10", 743.01],
                            ["S2", "2020-08-01 00:01:24", 751.92],
                            ["S2", "2020-09-01 00:02:10", 761.10],
                            ["S2", "2020-09-01 00:20:42", 762.33],
                        ],
                    },
                },
                "trade_pr",
            ),
            (
                "simple_ts_no_series",
                {
                    "ts_idx": {"ts_col": "event_ts", "series_ids": []},
                    "df": {
                        "schema": "event_ts string, new_col float",
                        "ts_convert": ["event_ts"],
                        "data": [
                            ["2020-08-01 00:00:10", 349.21],
                            ["2020-08-01 00:01:10", 743.01],
                            ["2020-08-01 00:01:12", 351.32],
                            ["2020-08-01 00:01:24", 751.92],
                            ["2020-09-01 00:02:10", 361.1],
                            ["2020-09-01 00:19:12", 362.1],
                            ["2020-09-01 00:20:42", 762.33],
                        ],
                    },
                },
                "trade_pr",
            ),
            (
                "simple_date_idx",
                {
                    "ts_idx": {"ts_col": "date", "series_ids": ["new_col"]},
                    "df": {
                        "schema": "new_col string, date string, temp float",
                        "date_convert": ["date"],
                        "data": [
                            ["LGA", "2020-08-01", 27.58],
                            ["LGA", "2020-08-02", 28.79],
                            ["LGA", "2020-08-03", 28.53],
                            ["LGA", "2020-08-04", 25.57],
                            ["YYZ", "2020-08-01", 24.16],
                            ["YYZ", "2020-08-02", 22.25],
                            ["YYZ", "2020-08-03", 20.62],
                            ["YYZ", "2020-08-04", 20.65],
                        ],
                    },
                },
                "station",
            ),
            (
                "ordinal_double_index",
                {
                    "ts_idx": {"ts_col": "event_ts_dbl", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, event_ts_dbl double, new_col float",
                        "data": [
                            ["S1", 0.13, 349.21],
                            ["S1", 1.207, 351.32],
                            ["S1", 10.0, 361.1],
                            ["S1", 24.357, 362.1],
                            ["S2", 0.005, 743.01],
                            ["S2", 0.1, 751.92],
                            ["S2", 1.0, 761.10],
                            ["S2", 10.0, 762.33],
                        ],
                    },
                },
                "trade_pr",
            ),
            (
                "ordinal_int_index",
                {
                    "ts_idx": {"ts_col": "order", "series_ids": ["new_col"]},
                    "df": {
                        "schema": "new_col string, order int, trade_pr float",
                        "data": [
                            ["S1", 1, 349.21],
                            ["S1", 20, 351.32],
                            ["S1", 127, 361.1],
                            ["S1", 243, 362.1],
                            ["S2", 0, 743.01],
                            ["S2", 1, 751.92],
                            ["S2", 10, 761.10],
                            ["S2", 100, 762.33],
                        ],
                    },
                },
                "symbol",
            ),
            (
                "parsed_ts_idx",
                {
                    "ts_idx": {
                        "ts_col": "event_ts",
                        "series_ids": ["symbol"],
                        "ts_fmt": "yyyy-MM-dd HH:mm:ss.SSS",
                    },
                    "tsdf_constructor": "fromStringTimestamp",
                    "df": {
                        "schema": "symbol string, event_ts string, new_col float",
                        "data": [
                            ["S1", "2020-08-01 00:00:10.010", 349.21],
                            ["S1", "2020-08-01 00:01:12.021", 351.32],
                            ["S1", "2020-09-01 00:02:10.032", 361.1],
                            ["S1", "2020-09-01 00:19:12.043", 362.1],
                            ["S2", "2020-08-01 00:01:10.054", 743.01],
                            ["S2", "2020-08-01 00:01:24.065", 751.92],
                            ["S2", "2020-09-01 00:02:10.076", 761.10],
                            ["S2", "2020-09-01 00:20:42.087", 762.33],
                        ],
                    },
                },
                "trade_pr",
            ),
            (
                "parsed_date_idx",
                {
                    "ts_idx": {
                        "ts_col": "date",
                        "series_ids": ["new_col"],
                        "ts_fmt": "yyyy-MM-dd",
                    },
                    "tsdf_constructor": "fromStringTimestamp",
                    "df": {
                        "schema": "new_col string, date string, temp float",
                        "data": [
                            ["LGA", "2020-08-01", 27.58],
                            ["LGA", "2020-08-02", 28.79],
                            ["LGA", "2020-08-03", 28.53],
                            ["LGA", "2020-08-04", 25.57],
                            ["YYZ", "2020-08-01", 24.16],
                            ["YYZ", "2020-08-02", 22.25],
                            ["YYZ", "2020-08-03", 20.62],
                            ["YYZ", "2020-08-04", 20.65],
                        ],
                    },
                },
                "station",
            ),
        ]
    )
    def test_withColumnRenamed(self, init_tsdf_id, expected_tsdf_dict, old_col):
        # load TSDF
        tsdf = self.get_test_data(init_tsdf_id).as_tsdf()
        # rename column
        newcol_tsdf = tsdf.withColumnRenamed(old_col, "new_col")
        # validate the renamed column
        expected_tsdf = TestDataFrame(self.spark, expected_tsdf_dict).as_tsdf()
        self.assertDataFrameEquality(newcol_tsdf, expected_tsdf)

    @parameterized.expand(
        [
            ("simple_ts_idx", "event_ts", {"new_col", "symbol", "trade_pr"}),
            ("simple_ts_no_series", "event_ts", {"new_col", "trade_pr"}),
            ("simple_date_idx", "date", {"new_col", "station", "temp"}),
            ("ordinal_double_index", "event_ts_dbl", {"new_col", "symbol", "trade_pr"}),
            ("ordinal_int_index", "order", {"new_col", "symbol", "trade_pr"}),
            ("parsed_ts_idx", "ts_idx", {"new_col", "symbol", "trade_pr"}),
            ("parsed_date_idx", "ts_idx", {"new_col", "station", "temp"}),
        ]
    )
    def test_withColumnRenamed_tsidx(
        self, init_tsdf_id, old_ts_col, expected_tsdf_cols
    ):
        # load TSDF
        tsdf = self.get_test_data(init_tsdf_id).as_tsdf()
        # select column
        selected_tsdf = tsdf.withColumnRenamed(old_ts_col, "new_col")
        # validate selected column
        self.assertEqual(set(selected_tsdf.df.columns), expected_tsdf_cols)

    @parameterized.expand(
        [
            ("simple_ts_idx", "trade_pr", {"event_ts", "symbol"}),
            # ("simple_ts_no_series", "", {"event_ts", "trade_pr"}),
            ("simple_date_idx", "temp", {"date", "station"}),
            ("ordinal_double_index", "trade_pr", {"event_ts_dbl", "symbol"}),
            ("ordinal_int_index", "trade_pr", {"order", "symbol"}),
            ("parsed_ts_idx", "trade_pr", {"ts_idx", "symbol"}),
            ("parsed_date_idx", "temp", {"ts_idx", "station"}),
        ]
    )
    def test_drop(self, init_tsdf_id, col_to_drop, expected_tsdf_cols):
        # load TSDF
        tsdf = self.get_test_data(init_tsdf_id).as_tsdf()
        # select column
        dropped_tsdf = tsdf.drop(col_to_drop)
        # validate selected column
        self.assertEqual(set(dropped_tsdf.df.columns), expected_tsdf_cols)

    @parameterized.expand(
        [
            ("simple_ts_idx", "event_ts"),
            ("simple_ts_no_series", "event_ts"),
            ("simple_date_idx", "date"),
            ("ordinal_double_index", "event_ts_dbl"),
            ("ordinal_int_index", "order"),
            ("parsed_ts_idx", "ts_idx"),
            ("parsed_date_idx", "ts_idx"),
        ]
    )
    def test_drop_tsidx(self, init_tsdf_id, col_to_drop):
        # load TSDF
        tsdf = self.get_test_data(init_tsdf_id).as_tsdf()
        # validate dropping the structural column throws an error
        self.assertRaises(AssertionError, tsdf.drop, col_to_drop)

    @parameterized.expand(
        [
            (
                "simple_ts_idx",
                "trade_pr",
                "string",
                StructType(
                    [
                        StructField("symbol", StringType(), True),
                        StructField("event_ts", TimestampType(), True),
                        StructField("trade_pr", StringType(), True),
                    ]
                ),
            ),
            (
                "simple_ts_no_series",
                "trade_pr",
                "string",
                StructType(
                    [
                        StructField("event_ts", TimestampType(), True),
                        StructField("trade_pr", StringType(), True),
                    ]
                ),
            ),
            (
                "simple_date_idx",
                "temp",
                "string",
                StructType(
                    [
                        StructField("station", StringType(), True),
                        StructField("date", DateType(), True),
                        StructField("temp", StringType(), True),
                    ]
                ),
            ),
            (
                "ordinal_double_index",
                "trade_pr",
                "string",
                StructType(
                    [
                        StructField("symbol", StringType(), True),
                        StructField("event_ts_dbl", DoubleType(), True),
                        StructField("trade_pr", StringType(), True),
                    ]
                ),
            ),
            (
                "ordinal_int_index",
                "trade_pr",
                "string",
                StructType(
                    [
                        StructField("symbol", StringType(), True),
                        StructField("order", IntegerType(), True),
                        StructField("trade_pr", StringType(), True),
                    ]
                ),
            ),
            (
                "parsed_ts_idx",
                "trade_pr",
                "string",
                StructType(
                    [
                        StructField("symbol", StringType(), True),
                        StructField("trade_pr", StringType(), True),
                        StructField(
                            "ts_idx",
                            StructType(
                                [
                                    StructField("event_ts", StringType(), True),
                                    StructField("parsed_ts", TimestampType(), True),
                                ]
                            ),
                            False,
                        ),
                    ]
                ),
            ),
            (
                "parsed_date_idx",
                "temp",
                "string",
                StructType(
                    [
                        StructField("station", StringType(), True),
                        StructField("temp", StringType(), True),
                        StructField(
                            "ts_idx",
                            StructType(
                                [
                                    StructField("date", StringType(), True),
                                    StructField("parsed_ts", DateType(), True),
                                ]
                            ),
                            False,
                        ),
                    ]
                ),
            ),
        ]
    )
    def test_withColumnTypeChanged(
        self, init_tsdf_id, col_to_change, new_type, expected_schema
    ):
        # load TSDF
        tsdf = self.get_test_data(init_tsdf_id).as_tsdf()
        # select column
        changed_tsdf = tsdf.withColumnTypeChanged(col_to_change, new_type)
        # validate selected column
        self.assertEqual(changed_tsdf.df.schema, expected_schema)

    # TODO: ts_index is expecting a timestamp not a string, should we check for this error or change that expectation?
    # @parameterized.expand([
    #     ("simple_ts_idx", "event_ts", "string",
    #      StructType([StructField('symbol', StringType(), True),
    #                  StructField('event_ts', StringType(), True),
    #                  StructField('trade_pr', StringType(), True)])),
    #     ("simple_ts_no_series", "trade_pr", "string",
    #      StructType([StructField('event_ts', TimestampType(), True),
    #                  StructField('trade_pr', StringType(), True)])),
    #     ("simple_date_idx", "temp", "string",
    #      StructType([StructField('station', StringType(), True),
    #                  StructField('date', DateType(), True),
    #                  StructField('temp', StringType(), True)])),
    #     ("ordinal_double_index", "trade_pr", "string",
    #      StructType([StructField('symbol', StringType(), True),
    #                  StructField('event_ts_dbl', DoubleType(), True),
    #                  StructField('trade_pr', StringType(), True)])),
    #     ("ordinal_int_index", "trade_pr", "string",
    #      StructType([StructField('symbol', StringType(), True),
    #                  StructField('order', IntegerType(), True),
    #                  StructField('trade_pr', StringType(), True)])),
    #     ("parsed_ts_idx", "trade_pr", "string",
    #      StructType([StructField('symbol', StringType(), True),
    #                  StructField('trade_pr', StringType(), True),
    #                 StructField('ts_idx', StructType([StructField('event_ts', StringType(), True),
    #                                                   StructField('parsed_ts', TimestampType(), True)]), False)])),
    #     ("parsed_date_idx", "temp", "string",
    #      StructType([StructField('station', StringType(), True),
    #                  StructField('temp', StringType(), True),
    #                 StructField('ts_idx', StructType([StructField('date', StringType(), True),
    #                                                   StructField('parsed_ts', DateType(), True)]), False)])),
    # ])
    # def test_withColumnTypeChanged_tsidx(self, init_tsdf_id, col_to_change, new_type, expected_schema):
    #     # load TSDF
    #     tsdf = self.get_test_data(init_tsdf_id).as_tsdf()
    #     # select column
    #     changed_tsdf = tsdf.withColumnTypeChanged(col_to_change, new_type)
    #     # validate selected column
    #     self.assertEqual(changed_tsdf.ts_schema, expected_schema)

    # TODO: AssertionError is thrown for inequal dataframes and I can't find a single character different in schema or data
    # @parameterized.expand([
    #     (
    #         "simple_ts_idx",
    #         "symbol string, event_ts timestamp, trade_pr float, new_col int",
    #         {
    #             "ts_idx": {"ts_col": "event_ts", "series_ids": ["symbol"]},
    #             "df": {
    #                 "schema": "symbol string, event_ts string, trade_pr float, new_col int",
    #                 "ts_convert": ["event_ts"],
    #                 "data": [
    #                     ["S1", "2020-08-01 00:00:10", 349.21, 1],
    #                     ["S1", "2020-08-01 00:01:12", 351.32, 1],
    #                     ["S1", "2020-09-01 00:02:10", 361.1, 1],
    #                     ["S1", "2020-09-01 00:19:12", 362.1, 1],
    #                     ["S2", "2020-08-01 00:01:10", 743.01, 1],
    #                     ["S2", "2020-08-01 00:01:24", 751.92, 1],
    #                     ["S2", "2020-09-01 00:02:10", 761.10, 1],
    #                     ["S2", "2020-09-01 00:20:42", 762.33, 1]
    #                 ]
    #             },
    #         },
    #     ),
    #     (
    #         "simple_ts_no_series",
    #         "event_ts string, trade_pr float, new_col int",
    #         {
    #             "ts_idx": {"ts_col": "event_ts", "series_ids": []},
    #             "df": {
    #                 "schema": "event_ts string, trade_pr float, new_col int",
    #                 "ts_convert": ["event_ts"],
    #                 "data": [
    #                     ["2020-08-01 00:00:10", 349.21, 1],
    #                     ["2020-08-01 00:01:10", 743.01, 1],
    #                     ["2020-08-01 00:01:12", 351.32, 1],
    #                     ["2020-08-01 00:01:24", 751.92, 1],
    #                     ["2020-09-01 00:02:10", 361.1, 1],
    #                     ["2020-09-01 00:19:12", 362.1, 1],
    #                     ["2020-09-01 00:20:42", 762.33, 1]
    #                 ],
    #             },
    #         },
    #     ),
    #     (
    #         "simple_date_idx",
    #         "station string, date string, temp float, new_col int",
    #         {
    #             "ts_idx": {"ts_col": "date", "series_ids": ["station"]},
    #             "df": {
    #                 "schema": "station string, date string, temp float, new_col int",
    #                 "date_convert": ["date"],
    #                 "data": [
    #                     ["LGA", "2020-08-01", 27.58, 1],
    #                     ["LGA", "2020-08-02", 28.79, 1],
    #                     ["LGA", "2020-08-03", 28.53, 1],
    #                     ["LGA", "2020-08-04", 25.57, 1],
    #                     ["YYZ", "2020-08-01", 24.16, 1],
    #                     ["YYZ", "2020-08-02", 22.25, 1],
    #                     ["YYZ", "2020-08-03", 20.62, 1],
    #                     ["YYZ", "2020-08-04", 20.65, 1]
    #                 ],
    #             },
    #         },
    #     ),
    #     (
    #         "ordinal_double_index",
    #         "symbol string, event_ts_dbl double, trade_pr float, new_col int",
    #         {
    #             "ts_idx": {"ts_col": "event_ts_dbl", "series_ids": ["symbol"]},
    #             "df": {
    #                 "schema": "symbol string, event_ts_dbl double, trade_pr float, new_col int",
    #                 "data": [
    #                     ["S1", 0.13, 349.21, 1],
    #                     ["S1", 1.207, 351.32, 1],
    #                     ["S1", 10.0, 361.1, 1],
    #                     ["S1", 24.357, 362.1, 1],
    #                     ["S2", 0.005, 743.01, 1],
    #                     ["S2", 0.1, 751.92, 1],
    #                     ["S2", 1.0, 761.10, 1],
    #                     ["S2", 10.0, 762.33, 1]
    #                 ]
    #             },
    #         },
    #     ),
    #     (
    #         "ordinal_int_index",
    #         "symbol string, order int, trade_pr float, new_col int",
    #         {
    #             "ts_idx": {"ts_col": "order", "series_ids": ["symbol"]},
    #             "df": {
    #                 "schema": "symbol string, order int, trade_pr float, new_col int",
    #                 "data": [
    #                     ["S1", 1, 349.21, 1],
    #                     ["S1", 20, 351.32, 1],
    #                     ["S1", 127, 361.1, 1],
    #                     ["S1", 243, 362.1, 1],
    #                     ["S2", 0, 743.01, 1],
    #                     ["S2", 1, 751.92, 1],
    #                     ["S2", 10, 761.10, 1],
    #                     ["S2", 100, 762.33, 1]
    #                 ]
    #             },
    #         },
    #     ),
    #     (
    #         "parsed_ts_idx",
    #         "symbol string, event_ts string, trade_pr float, new_col int",
    #         {
    #             "ts_idx": {
    #                 "ts_col": "event_ts",
    #                 "series_ids": ["symbol"],
    #                 "ts_fmt": "yyyy-MM-dd HH:mm:ss.SSS",
    #             },
    #             "tsdf_constructor": "fromStringTimestamp",
    #             "df": {
    #                 "schema": "symbol string, event_ts string, trade_pr float, new_col int",
    #                 "data": [
    #                     ["S1", "2020-08-01 00:00:10.010", 349.21, 1],
    #                     ["S1", "2020-08-01 00:01:12.021", 351.32, 1],
    #                     ["S1", "2020-09-01 00:02:10.032", 361.1, 1],
    #                     ["S1", "2020-09-01 00:19:12.043", 362.1, 1],
    #                     ["S2", "2020-08-01 00:01:10.054", 743.01, 1],
    #                     ["S2", "2020-08-01 00:01:24.065", 751.92, 1],
    #                     ["S2", "2020-09-01 00:02:10.076", 761.10, 1],
    #                     ["S2", "2020-09-01 00:20:42.087", 762.33, 1]
    #                 ]
    #             },
    #         },
    #     ),
    #     (
    #         "parsed_date_idx",
    #         "station string, date string, temp float, new_col int",
    #         {
    #             "ts_idx": {
    #                 "ts_col": "date",
    #                 "series_ids": ["station"],
    #                 "ts_fmt": "yyyy-MM-dd",
    #             },
    #             "tsdf_constructor": "fromStringTimestamp",
    #             "df": {
    #                 "schema": "station string, date string, temp float, new_col int",
    #                 "data": [
    #                     ["LGA", "2020-08-01", 27.58, 1],
    #                     ["LGA", "2020-08-02", 28.79, 1],
    #                     ["LGA", "2020-08-03", 28.53, 1],
    #                     ["LGA", "2020-08-04", 25.57, 1],
    #                     ["YYZ", "2020-08-01", 24.16, 1],
    #                     ["YYZ", "2020-08-02", 22.25, 1],
    #                     ["YYZ", "2020-08-03", 20.62, 1],
    #                     ["YYZ", "2020-08-04", 20.65, 1]
    #                 ],
    #             },
    #         },
    #     )
    # ])
    # def test_mapInPandas(self, init_tsdf_id, pandas_schema, expected_tsdf_dict):
    #     # load TSDF
    #     tsdf = self.get_test_data(init_tsdf_id).as_tsdf()
    #     # define pandas function to map
    #     def pandas_func(iterator):
    #         for pdf in iterator:
    #             pdf["new_col"] = 1
    #             yield pdf
    #     # select column
    #     pandas_result_tsdf = tsdf.mapInPandas(pandas_func, pandas_schema)
    #     # validate selected column
    #     expected_tsdf = TestDataFrame(self.spark, expected_tsdf_dict).as_tsdf()
    #     self.assertEqual(pandas_result_tsdf, expected_tsdf)

    @parameterized.expand(
        [
            (
                "simple_ts_idx",
                {
                    "ts_idx": {"ts_col": "event_ts", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, event_ts string, trade_pr float",
                        "ts_convert": ["event_ts"],
                        "data": [
                            ["S1", "2020-08-01 00:00:10", 349.21],
                            ["S1", "2020-08-01 00:01:12", 351.32],
                            ["S1", "2020-09-01 00:02:10", 361.1],
                            ["S1", "2020-09-01 00:19:12", 362.1],
                        ],
                    },
                },
                {
                    "ts_idx": {"ts_col": "event_ts", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, event_ts string, trade_pr float",
                        "ts_convert": ["event_ts"],
                        "data": [
                            ["S2", "2020-08-01 00:01:10", 743.01],
                            ["S2", "2020-08-01 00:01:24", 751.92],
                            ["S2", "2020-09-01 00:02:10", 761.10],
                            ["S2", "2020-09-01 00:20:42", 762.33],
                        ],
                    },
                },
            ),
            (
                "simple_ts_no_series",
                {
                    "ts_idx": {"ts_col": "event_ts", "series_ids": []},
                    "df": {
                        "schema": "event_ts string, trade_pr float",
                        "ts_convert": ["event_ts"],
                        "data": [
                            ["2020-08-01 00:00:10", 349.21],
                            ["2020-08-01 00:01:10", 743.01],
                            ["2020-08-01 00:01:12", 351.32],
                            ["2020-08-01 00:01:24", 751.92],
                        ],
                    },
                },
                {
                    "ts_idx": {"ts_col": "event_ts", "series_ids": []},
                    "df": {
                        "schema": "event_ts string, trade_pr float",
                        "ts_convert": ["event_ts"],
                        "data": [
                            ["2020-09-01 00:02:10", 361.1],
                            ["2020-09-01 00:19:12", 362.1],
                            ["2020-09-01 00:20:42", 762.33],
                        ],
                    },
                },
            ),
            (
                "simple_date_idx",
                {
                    "ts_idx": {"ts_col": "date", "series_ids": ["station"]},
                    "df": {
                        "schema": "station string, date string, temp float",
                        "date_convert": ["date"],
                        "data": [
                            ["LGA", "2020-08-01", 27.58],
                            ["LGA", "2020-08-02", 28.79],
                            ["LGA", "2020-08-03", 28.53],
                            ["LGA", "2020-08-04", 25.57],
                        ],
                    },
                },
                {
                    "ts_idx": {"ts_col": "date", "series_ids": ["station"]},
                    "df": {
                        "schema": "station string, date string, temp float",
                        "date_convert": ["date"],
                        "data": [
                            ["YYZ", "2020-08-01", 24.16],
                            ["YYZ", "2020-08-02", 22.25],
                            ["YYZ", "2020-08-03", 20.62],
                            ["YYZ", "2020-08-04", 20.65],
                        ],
                    },
                },
            ),
            (
                "ordinal_double_index",
                {
                    "ts_idx": {"ts_col": "event_ts_dbl", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, event_ts_dbl double, trade_pr float",
                        "data": [
                            ["S1", 0.13, 349.21],
                            ["S1", 1.207, 351.32],
                            ["S1", 10.0, 361.1],
                            ["S1", 24.357, 362.1],
                        ],
                    },
                },
                {
                    "ts_idx": {"ts_col": "event_ts_dbl", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, event_ts_dbl double, trade_pr float",
                        "data": [
                            ["S2", 0.005, 743.01],
                            ["S2", 0.1, 751.92],
                            ["S2", 1.0, 761.10],
                            ["S2", 10.0, 762.33],
                        ],
                    },
                },
            ),
            (
                "ordinal_int_index",
                {
                    "ts_idx": {"ts_col": "order", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, order int, trade_pr float",
                        "data": [
                            ["S1", 1, 349.21],
                            ["S1", 20, 351.32],
                            ["S1", 127, 361.1],
                            ["S1", 243, 362.1],
                        ],
                    },
                },
                {
                    "ts_idx": {"ts_col": "order", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, order int, trade_pr float",
                        "data": [
                            ["S2", 0, 743.01],
                            ["S2", 1, 751.92],
                            ["S2", 10, 761.10],
                            ["S2", 100, 762.33],
                        ],
                    },
                },
            ),
            (
                "parsed_ts_idx",
                {
                    "ts_idx": {
                        "ts_col": "event_ts",
                        "series_ids": ["symbol"],
                        "ts_fmt": "yyyy-MM-dd HH:mm:ss.SSS",
                    },
                    "tsdf_constructor": "fromStringTimestamp",
                    "df": {
                        "schema": "symbol string, event_ts string, trade_pr float",
                        "data": [
                            ["S1", "2020-08-01 00:00:10.010", 349.21],
                            ["S1", "2020-08-01 00:01:12.021", 351.32],
                            ["S1", "2020-09-01 00:02:10.032", 361.1],
                            ["S1", "2020-09-01 00:19:12.043", 362.1],
                        ],
                    },
                },
                {
                    "ts_idx": {
                        "ts_col": "event_ts",
                        "series_ids": ["symbol"],
                        "ts_fmt": "yyyy-MM-dd HH:mm:ss.SSS",
                    },
                    "tsdf_constructor": "fromStringTimestamp",
                    "df": {
                        "schema": "symbol string, event_ts string, trade_pr float",
                        "data": [
                            ["S2", "2020-08-01 00:01:10.054", 743.01],
                            ["S2", "2020-08-01 00:01:24.065", 751.92],
                            ["S2", "2020-09-01 00:02:10.076", 761.10],
                            ["S2", "2020-09-01 00:20:42.087", 762.33],
                        ],
                    },
                },
            ),
            (
                "parsed_date_idx",
                {
                    "ts_idx": {
                        "ts_col": "date",
                        "series_ids": ["station"],
                        "ts_fmt": "yyyy-MM-dd",
                    },
                    "tsdf_constructor": "fromStringTimestamp",
                    "df": {
                        "schema": "station string, date string, temp float",
                        "data": [
                            ["LGA", "2020-08-01", 27.58],
                            ["LGA", "2020-08-02", 28.79],
                            ["LGA", "2020-08-03", 28.53],
                            ["LGA", "2020-08-04", 25.57],
                        ],
                    },
                },
                {
                    "ts_idx": {
                        "ts_col": "date",
                        "series_ids": ["station"],
                        "ts_fmt": "yyyy-MM-dd",
                    },
                    "tsdf_constructor": "fromStringTimestamp",
                    "df": {
                        "schema": "station string, date string, temp float",
                        "data": [
                            ["YYZ", "2020-08-01", 24.16],
                            ["YYZ", "2020-08-02", 22.25],
                            ["YYZ", "2020-08-03", 20.62],
                            ["YYZ", "2020-08-04", 20.65],
                        ],
                    },
                },
            ),
        ]
    )
    def test_union(self, init_tsdf_id, tsdf_one_dict, tsdf_two_dict):
        # load expected TSDF
        expected_tsdf = self.get_test_data(init_tsdf_id).as_tsdf()
        # set up both TSDFs
        tsdf_one = TestDataFrame(self.spark, tsdf_one_dict).as_tsdf()
        tsdf_two = TestDataFrame(self.spark, tsdf_two_dict).as_tsdf()
        # add column
        unioned_tsdf = tsdf_one.union(tsdf_two)
        self.assertDataFrameEquality(unioned_tsdf, expected_tsdf)

    @parameterized.expand(
        [
            (
                "simple_ts_idx",
                {
                    "ts_idx": {"ts_col": "event_ts", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, trade_pr float, event_ts string",
                        "ts_convert": ["event_ts"],
                        "data": [
                            ["S1", 349.21, "2020-08-01 00:00:10"],
                            ["S1", 351.32, "2020-08-01 00:01:12"],
                            ["S1", 361.1, "2020-09-01 00:02:10"],
                            ["S1", 362.1, "2020-09-01 00:19:12"],
                        ],
                    },
                },
                {
                    "ts_idx": {"ts_col": "event_ts", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, event_ts string, trade_pr float",
                        "ts_convert": ["event_ts"],
                        "data": [
                            ["S2", "2020-08-01 00:01:10", 743.01],
                            ["S2", "2020-08-01 00:01:24", 751.92],
                            ["S2", "2020-09-01 00:02:10", 761.10],
                            ["S2", "2020-09-01 00:20:42", 762.33],
                        ],
                    },
                },
            ),
            (
                "simple_ts_no_series",
                {
                    "ts_idx": {"ts_col": "event_ts", "series_ids": []},
                    "df": {
                        "schema": "trade_pr float, event_ts string",
                        "ts_convert": ["event_ts"],
                        "data": [
                            [349.21, "2020-08-01 00:00:10"],
                            [743.01, "2020-08-01 00:01:10"],
                            [351.32, "2020-08-01 00:01:12"],
                            [751.92, "2020-08-01 00:01:24"],
                        ],
                    },
                },
                {
                    "ts_idx": {"ts_col": "event_ts", "series_ids": []},
                    "df": {
                        "schema": "event_ts string, trade_pr float",
                        "ts_convert": ["event_ts"],
                        "data": [
                            ["2020-09-01 00:02:10", 361.1],
                            ["2020-09-01 00:19:12", 362.1],
                            ["2020-09-01 00:20:42", 762.33],
                        ],
                    },
                },
            ),
            (
                "simple_date_idx",
                {
                    "ts_idx": {"ts_col": "date", "series_ids": ["station"]},
                    "df": {
                        "schema": "station string, temp float, date string",
                        "date_convert": ["date"],
                        "data": [
                            ["LGA", 27.58, "2020-08-01"],
                            ["LGA", 28.79, "2020-08-02"],
                            ["LGA", 28.53, "2020-08-03"],
                            ["LGA", 25.57, "2020-08-04"],
                        ],
                    },
                },
                {
                    "ts_idx": {"ts_col": "date", "series_ids": ["station"]},
                    "df": {
                        "schema": "station string, date string, temp float",
                        "date_convert": ["date"],
                        "data": [
                            ["YYZ", "2020-08-01", 24.16],
                            ["YYZ", "2020-08-02", 22.25],
                            ["YYZ", "2020-08-03", 20.62],
                            ["YYZ", "2020-08-04", 20.65],
                        ],
                    },
                },
            ),
            (
                "ordinal_double_index",
                {
                    "ts_idx": {"ts_col": "event_ts_dbl", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "event_ts_dbl double, trade_pr float, symbol string",
                        "data": [
                            [0.13, 349.21, "S1"],
                            [1.207, 351.32, "S1"],
                            [10.0, 361.1, "S1"],
                            [24.357, 362.1, "S1"],
                        ],
                    },
                },
                {
                    "ts_idx": {"ts_col": "event_ts_dbl", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, event_ts_dbl double, trade_pr float",
                        "data": [
                            ["S2", 0.005, 743.01],
                            ["S2", 0.1, 751.92],
                            ["S2", 1.0, 761.10],
                            ["S2", 10.0, 762.33],
                        ],
                    },
                },
            ),
            (
                "ordinal_int_index",
                {
                    "ts_idx": {"ts_col": "order", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, trade_pr float, order int",
                        "data": [
                            ["S1", 349.21, 1],
                            ["S1", 351.32, 20],
                            ["S1", 361.1, 127],
                            ["S1", 362.1, 243],
                        ],
                    },
                },
                {
                    "ts_idx": {"ts_col": "order", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, order int, trade_pr float",
                        "data": [
                            ["S2", 0, 743.01],
                            ["S2", 1, 751.92],
                            ["S2", 10, 761.10],
                            ["S2", 100, 762.33],
                        ],
                    },
                },
            ),
            (
                "parsed_ts_idx",
                {
                    "ts_idx": {
                        "ts_col": "event_ts",
                        "series_ids": ["symbol"],
                        "ts_fmt": "yyyy-MM-dd HH:mm:ss.SSS",
                    },
                    "tsdf_constructor": "fromStringTimestamp",
                    "df": {
                        "schema": "event_ts string, trade_pr float, symbol string",
                        "data": [
                            ["2020-08-01 00:00:10.010", 349.21, "S1"],
                            ["2020-08-01 00:01:12.021", 351.32, "S1"],
                            ["2020-09-01 00:02:10.032", 361.1, "S1"],
                            ["2020-09-01 00:19:12.043", 362.1, "S1"],
                        ],
                    },
                },
                {
                    "ts_idx": {
                        "ts_col": "event_ts",
                        "series_ids": ["symbol"],
                        "ts_fmt": "yyyy-MM-dd HH:mm:ss.SSS",
                    },
                    "tsdf_constructor": "fromStringTimestamp",
                    "df": {
                        "schema": "symbol string, event_ts string, trade_pr float",
                        "data": [
                            ["S2", "2020-08-01 00:01:10.054", 743.01],
                            ["S2", "2020-08-01 00:01:24.065", 751.92],
                            ["S2", "2020-09-01 00:02:10.076", 761.10],
                            ["S2", "2020-09-01 00:20:42.087", 762.33],
                        ],
                    },
                },
            ),
            (
                "parsed_date_idx",
                {
                    "ts_idx": {
                        "ts_col": "date",
                        "series_ids": ["station"],
                        "ts_fmt": "yyyy-MM-dd",
                    },
                    "tsdf_constructor": "fromStringTimestamp",
                    "df": {
                        "schema": "date string, temp float, station string",
                        "data": [
                            ["2020-08-01", 27.58, "LGA"],
                            ["2020-08-02", 28.79, "LGA"],
                            ["2020-08-03", 28.53, "LGA"],
                            ["2020-08-04", 25.57, "LGA"],
                        ],
                    },
                },
                {
                    "ts_idx": {
                        "ts_col": "date",
                        "series_ids": ["station"],
                        "ts_fmt": "yyyy-MM-dd",
                    },
                    "tsdf_constructor": "fromStringTimestamp",
                    "df": {
                        "schema": "station string, date string, temp float",
                        "data": [
                            ["YYZ", "2020-08-01", 24.16],
                            ["YYZ", "2020-08-02", 22.25],
                            ["YYZ", "2020-08-03", 20.62],
                            ["YYZ", "2020-08-04", 20.65],
                        ],
                    },
                },
            ),
        ]
    )
    def test_unionByNameNoMissing(self, init_tsdf_id, tsdf_one_dict, tsdf_two_dict):
        # load expected TSDF
        expected_tsdf = self.get_test_data(init_tsdf_id).as_tsdf()
        # set up both TSDFs
        tsdf_one = TestDataFrame(self.spark, tsdf_one_dict).as_tsdf()
        tsdf_two = TestDataFrame(self.spark, tsdf_two_dict).as_tsdf()
        # add column
        unioned_tsdf = tsdf_one.unionByName(tsdf_two)
        self.assertDataFrameEquality(unioned_tsdf, expected_tsdf)

    @parameterized.expand(
        [
            (
                "simple_ts_idx",
                {
                    "ts_idx": {"ts_col": "event_ts", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, event_ts string",
                        "ts_convert": ["event_ts"],
                        "data": [
                            ["S1", "2020-08-01 00:00:10"],
                            ["S1", "2020-08-01 00:01:12"],
                            ["S1", "2020-09-01 00:02:10"],
                            ["S1", "2020-09-01 00:19:12"],
                        ],
                    },
                },
                {
                    "ts_idx": {"ts_col": "event_ts", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, event_ts string, trade_pr float",
                        "ts_convert": ["event_ts"],
                        "data": [
                            ["S2", "2020-08-01 00:01:10", 743.01],
                            ["S2", "2020-08-01 00:01:24", 751.92],
                            ["S2", "2020-09-01 00:02:10", 761.10],
                            ["S2", "2020-09-01 00:20:42", 762.33],
                        ],
                    },
                },
            ),
            (
                "simple_ts_no_series",
                {
                    "ts_idx": {"ts_col": "event_ts", "series_ids": []},
                    "df": {
                        "schema": "event_ts string",
                        "ts_convert": ["event_ts"],
                        "data": [
                            ["2020-08-01 00:00:10"],
                            ["2020-08-01 00:01:10"],
                            ["2020-08-01 00:01:12"],
                            ["2020-08-01 00:01:24"],
                        ],
                    },
                },
                {
                    "ts_idx": {"ts_col": "event_ts", "series_ids": []},
                    "df": {
                        "schema": "event_ts string, trade_pr float",
                        "ts_convert": ["event_ts"],
                        "data": [
                            ["2020-09-01 00:02:10", 361.1],
                            ["2020-09-01 00:19:12", 362.1],
                            ["2020-09-01 00:20:42", 762.33],
                        ],
                    },
                },
            ),
            (
                "simple_date_idx",
                {
                    "ts_idx": {"ts_col": "date", "series_ids": ["station"]},
                    "df": {
                        "schema": "station string, date string",
                        "date_convert": ["date"],
                        "data": [
                            ["LGA", "2020-08-01"],
                            ["LGA", "2020-08-02"],
                            ["LGA", "2020-08-03"],
                            ["LGA", "2020-08-04"],
                        ],
                    },
                },
                {
                    "ts_idx": {"ts_col": "date", "series_ids": ["station"]},
                    "df": {
                        "schema": "station string, date string, temp float",
                        "date_convert": ["date"],
                        "data": [
                            ["YYZ", "2020-08-01", 24.16],
                            ["YYZ", "2020-08-02", 22.25],
                            ["YYZ", "2020-08-03", 20.62],
                            ["YYZ", "2020-08-04", 20.65],
                        ],
                    },
                },
            ),
            (
                "ordinal_double_index",
                {
                    "ts_idx": {"ts_col": "event_ts_dbl", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "event_ts_dbl double, symbol string",
                        "data": [
                            [0.13, "S1"],
                            [1.207, "S1"],
                            [10.0, "S1"],
                            [24.357, "S1"],
                        ],
                    },
                },
                {
                    "ts_idx": {"ts_col": "event_ts_dbl", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, event_ts_dbl double, trade_pr float",
                        "data": [
                            ["S2", 0.005, 743.01],
                            ["S2", 0.1, 751.92],
                            ["S2", 1.0, 761.10],
                            ["S2", 10.0, 762.33],
                        ],
                    },
                },
            ),
            (
                "ordinal_int_index",
                {
                    "ts_idx": {"ts_col": "order", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, order int",
                        "data": [["S1", 1], ["S1", 20], ["S1", 127], ["S1", 243]],
                    },
                },
                {
                    "ts_idx": {"ts_col": "order", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, order int, trade_pr float",
                        "data": [
                            ["S2", 0, 743.01],
                            ["S2", 1, 751.92],
                            ["S2", 10, 761.10],
                            ["S2", 100, 762.33],
                        ],
                    },
                },
            ),
            (
                "parsed_ts_idx",
                {
                    "ts_idx": {
                        "ts_col": "event_ts",
                        "series_ids": ["symbol"],
                        "ts_fmt": "yyyy-MM-dd HH:mm:ss.SSS",
                    },
                    "tsdf_constructor": "fromStringTimestamp",
                    "df": {
                        "schema": "event_ts string, symbol string",
                        "data": [
                            ["2020-08-01 00:00:10.010", "S1"],
                            ["2020-08-01 00:01:12.021", "S1"],
                            ["2020-09-01 00:02:10.032", "S1"],
                            ["2020-09-01 00:19:12.043", "S1"],
                        ],
                    },
                },
                {
                    "ts_idx": {
                        "ts_col": "event_ts",
                        "series_ids": ["symbol"],
                        "ts_fmt": "yyyy-MM-dd HH:mm:ss.SSS",
                    },
                    "tsdf_constructor": "fromStringTimestamp",
                    "df": {
                        "schema": "symbol string, event_ts string, trade_pr float",
                        "data": [
                            ["S2", "2020-08-01 00:01:10.054", 743.01],
                            ["S2", "2020-08-01 00:01:24.065", 751.92],
                            ["S2", "2020-09-01 00:02:10.076", 761.10],
                            ["S2", "2020-09-01 00:20:42.087", 762.33],
                        ],
                    },
                },
            ),
            (
                "parsed_date_idx",
                {
                    "ts_idx": {
                        "ts_col": "date",
                        "series_ids": ["station"],
                        "ts_fmt": "yyyy-MM-dd",
                    },
                    "tsdf_constructor": "fromStringTimestamp",
                    "df": {
                        "schema": "date string, station string",
                        "data": [
                            ["2020-08-01", "LGA"],
                            ["2020-08-02", "LGA"],
                            ["2020-08-03", "LGA"],
                            ["2020-08-04", "LGA"],
                        ],
                    },
                },
                {
                    "ts_idx": {
                        "ts_col": "date",
                        "series_ids": ["station"],
                        "ts_fmt": "yyyy-MM-dd",
                    },
                    "tsdf_constructor": "fromStringTimestamp",
                    "df": {
                        "schema": "station string, date string, temp float",
                        "data": [
                            ["YYZ", "2020-08-01", 24.16],
                            ["YYZ", "2020-08-02", 22.25],
                            ["YYZ", "2020-08-03", 20.62],
                            ["YYZ", "2020-08-04", 20.65],
                        ],
                    },
                },
            ),
        ]
    )
    def test_unionByNameMissing(self, init_tsdf_id, tsdf_one_dict, tsdf_two_dict):
        # load expected TSDF
        expected_tsdf = self.get_test_data(init_tsdf_id).as_tsdf()
        # set up both TSDFs
        tsdf_one = TestDataFrame(self.spark, tsdf_one_dict).as_tsdf()
        tsdf_two = TestDataFrame(self.spark, tsdf_two_dict).as_tsdf()
        # add column
        unioned_tsdf = tsdf_one.unionByName(tsdf_two, True)
        self.assertEqual(set(unioned_tsdf.df.columns), set(expected_tsdf.df.columns))


class TimeSlicingTests(SparkTest):
    @parameterized.expand(
        [
            (
                "simple_ts_idx",
                "2020-09-01 00:02:10",
                {
                    "ts_idx": {"ts_col": "event_ts", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, event_ts string, trade_pr float",
                        "ts_convert": ["event_ts"],
                        "data": [
                            ["S1", "2020-09-01 00:02:10", 361.1],
                            ["S2", "2020-09-01 00:02:10", 761.10],
                        ],
                    },
                },
            ),
            (
                "simple_ts_no_series",
                "2020-09-01 00:19:12",
                {
                    "ts_idx": {"ts_col": "event_ts", "series_ids": []},
                    "df": {
                        "schema": "event_ts string, trade_pr float",
                        "ts_convert": ["event_ts"],
                        "data": [
                            ["2020-09-01 00:19:12", 362.1],
                        ],
                    },
                },
            ),
            (
                "simple_date_idx",
                "2020-08-02",
                {
                    "ts_idx": {"ts_col": "date", "series_ids": ["station"]},
                    "df": {
                        "schema": "station string, date string, temp float",
                        "date_convert": ["date"],
                        "data": [
                            ["LGA", "2020-08-02", 28.79],
                            ["YYZ", "2020-08-02", 22.25],
                        ],
                    },
                },
            ),
            (
                "ordinal_double_index",
                10.0,
                {
                    "ts_idx": {"ts_col": "event_ts_dbl", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, event_ts_dbl double, trade_pr float",
                        "data": [
                            ["S1", 10.0, 361.1],
                            ["S2", 10.0, 762.33],
                        ],
                    },
                },
            ),
            (
                "ordinal_int_index",
                1,
                {
                    "ts_idx": {"ts_col": "order", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, order int, trade_pr float",
                        "data": [
                            ["S1", 1, 349.21],
                            ["S2", 1, 751.92],
                        ],
                    },
                },
            ),
            (
                "parsed_ts_idx",
                "2020-09-01 00:02:10.032",
                {
                    "ts_idx": {
                        "ts_col": "event_ts",
                        "series_ids": ["symbol"],
                        "ts_fmt": "yyyy-MM-dd HH:mm:ss.SSS",
                    },
                    "tsdf_constructor": "fromStringTimestamp",
                    "df": {
                        "schema": "symbol string, event_ts string, trade_pr float",
                        "data": [
                            ["S1", "2020-09-01 00:02:10.032", 361.1],
                        ],
                    },
                },
            ),
            (
                "parsed_date_idx",
                "2020-08-04",
                {
                    "ts_idx": {
                        "ts_col": "date",
                        "series_ids": ["station"],
                        "ts_fmt": "yyyy-MM-dd",
                    },
                    "tsdf_constructor": "fromStringTimestamp",
                    "df": {
                        "schema": "station string, date string, temp float",
                        "data": [
                            ["LGA", "2020-08-04", 25.57],
                            ["YYZ", "2020-08-04", 20.65],
                        ],
                    },
                },
            ),
        ]
    )
    def test_at(self, init_tsdf_id, ts, expected_tsdf_dict):
        # load TSDF
        tsdf = self.get_test_data(init_tsdf_id).as_tsdf()
        # slice at timestamp
        at_tsdf = tsdf.at(ts)
        # validate the slice
        expected_tsdf = TestDataFrame(self.spark, expected_tsdf_dict).as_tsdf()
        self.assertDataFrameEquality(at_tsdf, expected_tsdf)

    @parameterized.expand(
        [
            (
                "simple_ts_idx",
                "2020-09-01 00:02:10",
                {
                    "ts_idx": {"ts_col": "event_ts", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, event_ts string, trade_pr float",
                        "ts_convert": ["event_ts"],
                        "data": [
                            ["S1", "2020-08-01 00:00:10", 349.21],
                            ["S1", "2020-08-01 00:01:12", 351.32],
                            ["S2", "2020-08-01 00:01:10", 743.01],
                            ["S2", "2020-08-01 00:01:24", 751.92],
                        ],
                    },
                },
            ),
            (
                "simple_ts_no_series",
                "2020-09-01 00:19:12",
                {
                    "ts_idx": {"ts_col": "event_ts", "series_ids": []},
                    "df": {
                        "schema": "event_ts string, trade_pr float",
                        "ts_convert": ["event_ts"],
                        "data": [
                            ["2020-08-01 00:00:10", 349.21],
                            ["2020-08-01 00:01:10", 743.01],
                            ["2020-08-01 00:01:12", 351.32],
                            ["2020-08-01 00:01:24", 751.92],
                            ["2020-09-01 00:02:10", 361.1],
                        ],
                    },
                },
            ),
            (
                "simple_date_idx",
                "2020-08-03",
                {
                    "ts_idx": {"ts_col": "date", "series_ids": ["station"]},
                    "df": {
                        "schema": "station string, date string, temp float",
                        "date_convert": ["date"],
                        "data": [
                            ["LGA", "2020-08-01", 27.58],
                            ["LGA", "2020-08-02", 28.79],
                            ["YYZ", "2020-08-01", 24.16],
                            ["YYZ", "2020-08-02", 22.25],
                        ],
                    },
                },
            ),
            (
                "ordinal_double_index",
                10.0,
                {
                    "ts_idx": {"ts_col": "event_ts_dbl", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, event_ts_dbl double, trade_pr float",
                        "data": [
                            ["S1", 0.13, 349.21],
                            ["S1", 1.207, 351.32],
                            ["S2", 0.005, 743.01],
                            ["S2", 0.1, 751.92],
                            ["S2", 1.0, 761.10],
                        ],
                    },
                },
            ),
            (
                "ordinal_int_index",
                1,
                {
                    "ts_idx": {"ts_col": "order", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, order int, trade_pr float",
                        "data": [["S2", 0, 743.01]],
                    },
                },
            ),
            (
                "parsed_ts_idx",
                "2020-09-01 00:02:10.000",
                {
                    "ts_idx": {
                        "ts_col": "event_ts",
                        "series_ids": ["symbol"],
                        "ts_fmt": "yyyy-MM-dd HH:mm:ss.SSS",
                    },
                    "tsdf_constructor": "fromStringTimestamp",
                    "df": {
                        "schema": "symbol string, event_ts string, trade_pr float",
                        "data": [
                            ["S1", "2020-08-01 00:00:10.010", 349.21],
                            ["S1", "2020-08-01 00:01:12.021", 351.32],
                            ["S2", "2020-08-01 00:01:10.054", 743.01],
                            ["S2", "2020-08-01 00:01:24.065", 751.92],
                        ],
                    },
                },
            ),
            (
                "parsed_date_idx",
                "2020-08-03",
                {
                    "ts_idx": {
                        "ts_col": "date",
                        "series_ids": ["station"],
                        "ts_fmt": "yyyy-MM-dd",
                    },
                    "tsdf_constructor": "fromStringTimestamp",
                    "df": {
                        "schema": "station string, date string, temp float",
                        "data": [
                            ["LGA", "2020-08-01", 27.58],
                            ["LGA", "2020-08-02", 28.79],
                            ["YYZ", "2020-08-01", 24.16],
                            ["YYZ", "2020-08-02", 22.25],
                        ],
                    },
                },
            ),
        ]
    )
    def test_before(self, init_tsdf_id, ts, expected_tsdf_dict):
        # load TSDF
        tsdf = self.get_test_data(init_tsdf_id).as_tsdf()
        # slice at timestamp
        before_tsdf = tsdf.before(ts)
        # validate the slice
        expected_tsdf = TestDataFrame(self.spark, expected_tsdf_dict).as_tsdf()
        self.assertDataFrameEquality(before_tsdf, expected_tsdf)

    @parameterized.expand(
        [
            (
                "simple_ts_idx",
                "2020-09-01 00:02:10",
                {
                    "ts_idx": {"ts_col": "event_ts", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, event_ts string, trade_pr float",
                        "ts_convert": ["event_ts"],
                        "data": [
                            ["S1", "2020-08-01 00:00:10", 349.21],
                            ["S1", "2020-08-01 00:01:12", 351.32],
                            ["S1", "2020-09-01 00:02:10", 361.1],
                            ["S2", "2020-08-01 00:01:10", 743.01],
                            ["S2", "2020-08-01 00:01:24", 751.92],
                            ["S2", "2020-09-01 00:02:10", 761.10],
                        ],
                    },
                },
            ),
            (
                "simple_ts_no_series",
                "2020-09-01 00:19:12",
                {
                    "ts_idx": {"ts_col": "event_ts", "series_ids": []},
                    "df": {
                        "schema": "event_ts string, trade_pr float",
                        "ts_convert": ["event_ts"],
                        "data": [
                            ["2020-08-01 00:00:10", 349.21],
                            ["2020-08-01 00:01:10", 743.01],
                            ["2020-08-01 00:01:12", 351.32],
                            ["2020-08-01 00:01:24", 751.92],
                            ["2020-09-01 00:02:10", 361.1],
                            ["2020-09-01 00:19:12", 362.1],
                        ],
                    },
                },
            ),
            (
                "simple_date_idx",
                "2020-08-03",
                {
                    "ts_idx": {"ts_col": "date", "series_ids": ["station"]},
                    "df": {
                        "schema": "station string, date string, temp float",
                        "date_convert": ["date"],
                        "data": [
                            ["LGA", "2020-08-01", 27.58],
                            ["LGA", "2020-08-02", 28.79],
                            ["LGA", "2020-08-03", 28.53],
                            ["YYZ", "2020-08-01", 24.16],
                            ["YYZ", "2020-08-02", 22.25],
                            ["YYZ", "2020-08-03", 20.62],
                        ],
                    },
                },
            ),
            (
                "ordinal_double_index",
                10.0,
                {
                    "ts_idx": {"ts_col": "event_ts_dbl", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, event_ts_dbl double, trade_pr float",
                        "data": [
                            ["S1", 0.13, 349.21],
                            ["S1", 1.207, 351.32],
                            ["S1", 10.0, 361.1],
                            ["S2", 0.005, 743.01],
                            ["S2", 0.1, 751.92],
                            ["S2", 1.0, 761.10],
                            ["S2", 10.0, 762.33],
                        ],
                    },
                },
            ),
            (
                "ordinal_int_index",
                1,
                {
                    "ts_idx": {"ts_col": "order", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, order int, trade_pr float",
                        "data": [
                            ["S1", 1, 349.21],
                            ["S2", 0, 743.01],
                            ["S2", 1, 751.92],
                        ],
                    },
                },
            ),
            (
                "parsed_ts_idx",
                "2020-09-01 00:02:10.000",
                {
                    "ts_idx": {
                        "ts_col": "event_ts",
                        "series_ids": ["symbol"],
                        "ts_fmt": "yyyy-MM-dd HH:mm:ss.SSS",
                    },
                    "tsdf_constructor": "fromStringTimestamp",
                    "df": {
                        "schema": "symbol string, event_ts string, trade_pr float",
                        "data": [
                            ["S1", "2020-08-01 00:00:10.010", 349.21],
                            ["S1", "2020-08-01 00:01:12.021", 351.32],
                            ["S2", "2020-08-01 00:01:10.054", 743.01],
                            ["S2", "2020-08-01 00:01:24.065", 751.92],
                        ],
                    },
                },
            ),
            (
                "parsed_date_idx",
                "2020-08-03",
                {
                    "ts_idx": {
                        "ts_col": "date",
                        "series_ids": ["station"],
                        "ts_fmt": "yyyy-MM-dd",
                    },
                    "tsdf_constructor": "fromStringTimestamp",
                    "df": {
                        "schema": "station string, date string, temp float",
                        "data": [
                            ["LGA", "2020-08-01", 27.58],
                            ["LGA", "2020-08-02", 28.79],
                            ["LGA", "2020-08-03", 28.53],
                            ["YYZ", "2020-08-01", 24.16],
                            ["YYZ", "2020-08-02", 22.25],
                            ["YYZ", "2020-08-03", 20.62],
                        ],
                    },
                },
            ),
        ]
    )
    def test_atOrBefore(self, init_tsdf_id, ts, expected_tsdf_dict):
        # load TSDF
        tsdf = self.get_test_data(init_tsdf_id).as_tsdf()
        # slice at timestamp
        at_before_tsdf = tsdf.atOrBefore(ts)
        # validate the slice
        expected_tsdf = TestDataFrame(self.spark, expected_tsdf_dict).as_tsdf()
        self.assertDataFrameEquality(at_before_tsdf, expected_tsdf)

    @parameterized.expand(
        [
            (
                "simple_ts_idx",
                "2020-09-01 00:02:10",
                {
                    "ts_idx": {"ts_col": "event_ts", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, event_ts string, trade_pr float",
                        "ts_convert": ["event_ts"],
                        "data": [
                            ["S1", "2020-09-01 00:19:12", 362.1],
                            ["S2", "2020-09-01 00:20:42", 762.33],
                        ],
                    },
                },
            ),
            (
                "simple_ts_no_series",
                "2020-09-01 00:08:12",
                {
                    "ts_idx": {"ts_col": "event_ts", "series_ids": []},
                    "df": {
                        "schema": "event_ts string, trade_pr float",
                        "ts_convert": ["event_ts"],
                        "data": [
                            ["2020-09-01 00:19:12", 362.1],
                            ["2020-09-01 00:20:42", 762.33],
                        ],
                    },
                },
            ),
            (
                "simple_date_idx",
                "2020-08-02",
                {
                    "ts_idx": {"ts_col": "date", "series_ids": ["station"]},
                    "df": {
                        "schema": "station string, date string, temp float",
                        "date_convert": ["date"],
                        "data": [
                            ["LGA", "2020-08-03", 28.53],
                            ["LGA", "2020-08-04", 25.57],
                            ["YYZ", "2020-08-03", 20.62],
                            ["YYZ", "2020-08-04", 20.65],
                        ],
                    },
                },
            ),
            (
                "ordinal_double_index",
                1.0,
                {
                    "ts_idx": {"ts_col": "event_ts_dbl", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, event_ts_dbl double, trade_pr float",
                        "data": [
                            ["S1", 1.207, 351.32],
                            ["S1", 10.0, 361.1],
                            ["S1", 24.357, 362.1],
                            ["S2", 10.0, 762.33],
                        ],
                    },
                },
            ),
            (
                "ordinal_int_index",
                10,
                {
                    "ts_idx": {"ts_col": "order", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, order int, trade_pr float",
                        "data": [
                            ["S1", 20, 351.32],
                            ["S1", 127, 361.1],
                            ["S1", 243, 362.1],
                            ["S2", 100, 762.33],
                        ],
                    },
                },
            ),
            (
                "parsed_ts_idx",
                "2020-09-01 00:02:10.000",
                {
                    "ts_idx": {
                        "ts_col": "event_ts",
                        "series_ids": ["symbol"],
                        "ts_fmt": "yyyy-MM-dd HH:mm:ss.SSS",
                    },
                    "tsdf_constructor": "fromStringTimestamp",
                    "df": {
                        "schema": "symbol string, event_ts string, trade_pr float",
                        "data": [
                            ["S1", "2020-09-01 00:02:10.032", 361.1],
                            ["S1", "2020-09-01 00:19:12.043", 362.1],
                            ["S2", "2020-09-01 00:02:10.076", 761.10],
                            ["S2", "2020-09-01 00:20:42.087", 762.33],
                        ],
                    },
                },
            ),
            (
                "parsed_date_idx",
                "2020-08-03",
                {
                    "ts_idx": {
                        "ts_col": "date",
                        "series_ids": ["station"],
                        "ts_fmt": "yyyy-MM-dd",
                    },
                    "tsdf_constructor": "fromStringTimestamp",
                    "df": {
                        "schema": "station string, date string, temp float",
                        "data": [
                            ["LGA", "2020-08-04", 25.57],
                            ["YYZ", "2020-08-04", 20.65],
                        ],
                    },
                },
            ),
        ]
    )
    def test_after(self, init_tsdf_id, ts, expected_tsdf_dict):
        # load TSDF
        tsdf = self.get_test_data(init_tsdf_id).as_tsdf()
        # slice at timestamp
        after_tsdf = tsdf.after(ts)
        # validate the slice
        expected_tsdf = TestDataFrame(self.spark, expected_tsdf_dict).as_tsdf()
        self.assertDataFrameEquality(after_tsdf, expected_tsdf)

    @parameterized.expand(
        [
            (
                "simple_ts_idx",
                "2020-09-01 00:02:10",
                {
                    "ts_idx": {"ts_col": "event_ts", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, event_ts string, trade_pr float",
                        "ts_convert": ["event_ts"],
                        "data": [
                            ["S1", "2020-09-01 00:02:10", 361.1],
                            ["S1", "2020-09-01 00:19:12", 362.1],
                            ["S2", "2020-09-01 00:02:10", 761.10],
                            ["S2", "2020-09-01 00:20:42", 762.33],
                        ],
                    },
                },
            ),
            (
                "simple_ts_no_series",
                "2020-08-01 00:01:24",
                {
                    "ts_idx": {"ts_col": "event_ts", "series_ids": []},
                    "df": {
                        "schema": "event_ts string, trade_pr float",
                        "ts_convert": ["event_ts"],
                        "data": [
                            ["2020-08-01 00:01:24", 751.92],
                            ["2020-09-01 00:02:10", 361.1],
                            ["2020-09-01 00:19:12", 362.1],
                            ["2020-09-01 00:20:42", 762.33],
                        ],
                    },
                },
            ),
            (
                "simple_date_idx",
                "2020-08-03",
                {
                    "ts_idx": {"ts_col": "date", "series_ids": ["station"]},
                    "df": {
                        "schema": "station string, date string, temp float",
                        "date_convert": ["date"],
                        "data": [
                            ["LGA", "2020-08-03", 28.53],
                            ["LGA", "2020-08-04", 25.57],
                            ["YYZ", "2020-08-03", 20.62],
                            ["YYZ", "2020-08-04", 20.65],
                        ],
                    },
                },
            ),
            (
                "ordinal_double_index",
                10.0,
                {
                    "ts_idx": {"ts_col": "event_ts_dbl", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, event_ts_dbl double, trade_pr float",
                        "data": [
                            ["S1", 10.0, 361.1],
                            ["S1", 24.357, 362.1],
                            ["S2", 10.0, 762.33],
                        ],
                    },
                },
            ),
            (
                "ordinal_int_index",
                10,
                {
                    "ts_idx": {"ts_col": "order", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, order int, trade_pr float",
                        "data": [
                            ["S1", 20, 351.32],
                            ["S1", 127, 361.1],
                            ["S1", 243, 362.1],
                            ["S2", 10, 761.10],
                            ["S2", 100, 762.33],
                        ],
                    },
                },
            ),
            (
                "parsed_ts_idx",
                "2020-09-01 00:02:10.000",
                {
                    "ts_idx": {
                        "ts_col": "event_ts",
                        "series_ids": ["symbol"],
                        "ts_fmt": "yyyy-MM-dd HH:mm:ss.SSS",
                    },
                    "tsdf_constructor": "fromStringTimestamp",
                    "df": {
                        "schema": "symbol string, event_ts string, trade_pr float",
                        "data": [
                            ["S1", "2020-09-01 00:02:10.032", 361.1],
                            ["S1", "2020-09-01 00:19:12.043", 362.1],
                            ["S2", "2020-09-01 00:02:10.076", 761.10],
                            ["S2", "2020-09-01 00:20:42.087", 762.33],
                        ],
                    },
                },
            ),
            (
                "parsed_date_idx",
                "2020-08-03",
                {
                    "ts_idx": {
                        "ts_col": "date",
                        "series_ids": ["station"],
                        "ts_fmt": "yyyy-MM-dd",
                    },
                    "tsdf_constructor": "fromStringTimestamp",
                    "df": {
                        "schema": "station string, date string, temp float",
                        "data": [
                            ["LGA", "2020-08-03", 28.53],
                            ["LGA", "2020-08-04", 25.57],
                            ["YYZ", "2020-08-03", 20.62],
                            ["YYZ", "2020-08-04", 20.65],
                        ],
                    },
                },
            ),
        ]
    )
    def test_atOrAfter(self, init_tsdf_id, ts, expected_tsdf_dict):
        # load TSDF
        tsdf = self.get_test_data(init_tsdf_id).as_tsdf()
        # slice at timestamp
        at_after_tsdf = tsdf.atOrAfter(ts)
        # validate the slice
        expected_tsdf = TestDataFrame(self.spark, expected_tsdf_dict).as_tsdf()
        self.assertDataFrameEquality(at_after_tsdf, expected_tsdf)

    @parameterized.expand(
        [
            (
                "simple_ts_idx",
                "2020-08-01 00:01:10",
                "2020-09-01 00:02:10",
                {
                    "ts_idx": {"ts_col": "event_ts", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, event_ts string, trade_pr float",
                        "ts_convert": ["event_ts"],
                        "data": [
                            ["S1", "2020-08-01 00:01:12", 351.32],
                            ["S2", "2020-08-01 00:01:24", 751.92],
                        ],
                    },
                },
            ),
            (
                "simple_ts_no_series",
                "2020-08-01 00:01:10",
                "2020-09-01 00:02:10",
                {
                    "ts_idx": {"ts_col": "event_ts", "series_ids": []},
                    "df": {
                        "schema": "event_ts string, trade_pr float",
                        "ts_convert": ["event_ts"],
                        "data": [
                            ["2020-08-01 00:01:12", 351.32],
                            ["2020-08-01 00:01:24", 751.92],
                        ],
                    },
                },
            ),
            (
                "simple_date_idx",
                "2020-08-01",
                "2020-08-03",
                {
                    "ts_idx": {"ts_col": "date", "series_ids": ["station"]},
                    "df": {
                        "schema": "station string, date string, temp float",
                        "date_convert": ["date"],
                        "data": [
                            ["LGA", "2020-08-02", 28.79],
                            ["YYZ", "2020-08-02", 22.25],
                        ],
                    },
                },
            ),
            (
                "ordinal_double_index",
                0.1,
                10.0,
                {
                    "ts_idx": {"ts_col": "event_ts_dbl", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, event_ts_dbl double, trade_pr float",
                        "data": [
                            ["S1", 0.13, 349.21],
                            ["S1", 1.207, 351.32],
                            ["S2", 1.0, 761.10],
                        ],
                    },
                },
            ),
            (
                "ordinal_int_index",
                1,
                100,
                {
                    "ts_idx": {"ts_col": "order", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, order int, trade_pr float",
                        "data": [["S1", 20, 351.32], ["S2", 10, 761.10]],
                    },
                },
            ),
            (
                "parsed_ts_idx",
                "2020-08-01 00:00:10.010",
                "2020-09-01 00:02:10.076",
                {
                    "ts_idx": {
                        "ts_col": "event_ts",
                        "series_ids": ["symbol"],
                        "ts_fmt": "yyyy-MM-dd HH:mm:ss.SSS",
                    },
                    "tsdf_constructor": "fromStringTimestamp",
                    "df": {
                        "schema": "symbol string, event_ts string, trade_pr float",
                        "data": [
                            ["S1", "2020-08-01 00:01:12.021", 351.32],
                            ["S1", "2020-09-01 00:02:10.032", 361.1],
                            ["S2", "2020-08-01 00:01:10.054", 743.01],
                            ["S2", "2020-08-01 00:01:24.065", 751.92],
                        ],
                    },
                },
            ),
            (
                "parsed_date_idx",
                "2020-08-01",
                "2020-08-03",
                {
                    "ts_idx": {
                        "ts_col": "date",
                        "series_ids": ["station"],
                        "ts_fmt": "yyyy-MM-dd",
                    },
                    "tsdf_constructor": "fromStringTimestamp",
                    "df": {
                        "schema": "station string, date string, temp float",
                        "data": [
                            ["LGA", "2020-08-02", 28.79],
                            ["YYZ", "2020-08-02", 22.25],
                        ],
                    },
                },
            ),
        ]
    )
    def test_between_non_inclusive(
        self, init_tsdf_id, start_ts, end_ts, expected_tsdf_dict
    ):
        # load TSDF
        tsdf = self.get_test_data(init_tsdf_id).as_tsdf()
        # slice at timestamp
        between_tsdf = tsdf.between(start_ts, end_ts, inclusive=False)
        # validate the slice
        expected_tsdf = TestDataFrame(self.spark, expected_tsdf_dict).as_tsdf()
        self.assertDataFrameEquality(between_tsdf, expected_tsdf)

    @parameterized.expand(
        [
            (
                "simple_ts_idx",
                "2020-08-01 00:01:10",
                "2020-09-01 00:02:10",
                {
                    "ts_idx": {"ts_col": "event_ts", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, event_ts string, trade_pr float",
                        "ts_convert": ["event_ts"],
                        "data": [
                            ["S1", "2020-08-01 00:01:12", 351.32],
                            ["S1", "2020-09-01 00:02:10", 361.1],
                            ["S2", "2020-08-01 00:01:10", 743.01],
                            ["S2", "2020-08-01 00:01:24", 751.92],
                            ["S2", "2020-09-01 00:02:10", 761.10],
                        ],
                    },
                },
            ),
            (
                "simple_ts_no_series",
                "2020-08-01 00:01:10",
                "2020-09-01 00:02:10",
                {
                    "ts_idx": {"ts_col": "event_ts", "series_ids": []},
                    "df": {
                        "schema": "event_ts string, trade_pr float",
                        "ts_convert": ["event_ts"],
                        "data": [
                            ["2020-08-01 00:01:10", 743.01],
                            ["2020-08-01 00:01:12", 351.32],
                            ["2020-08-01 00:01:24", 751.92],
                            ["2020-09-01 00:02:10", 361.1],
                        ],
                    },
                },
            ),
            (
                "simple_date_idx",
                "2020-08-01",
                "2020-08-03",
                {
                    "ts_idx": {"ts_col": "date", "series_ids": ["station"]},
                    "df": {
                        "schema": "station string, date string, temp float",
                        "date_convert": ["date"],
                        "data": [
                            ["LGA", "2020-08-01", 27.58],
                            ["LGA", "2020-08-02", 28.79],
                            ["LGA", "2020-08-03", 28.53],
                            ["YYZ", "2020-08-01", 24.16],
                            ["YYZ", "2020-08-02", 22.25],
                            ["YYZ", "2020-08-03", 20.62],
                        ],
                    },
                },
            ),
            (
                "ordinal_double_index",
                0.1,
                10.0,
                {
                    "ts_idx": {"ts_col": "event_ts_dbl", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, event_ts_dbl double, trade_pr float",
                        "data": [
                            ["S1", 0.13, 349.21],
                            ["S1", 1.207, 351.32],
                            ["S1", 10.0, 361.1],
                            ["S2", 0.1, 751.92],
                            ["S2", 1.0, 761.10],
                            ["S2", 10.0, 762.33],
                        ],
                    },
                },
            ),
            (
                "ordinal_int_index",
                1,
                100,
                {
                    "ts_idx": {"ts_col": "order", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, order int, trade_pr float",
                        "data": [
                            ["S1", 1, 349.21],
                            ["S1", 20, 351.32],
                            ["S2", 1, 751.92],
                            ["S2", 10, 761.10],
                            ["S2", 100, 762.33],
                        ],
                    },
                },
            ),
            (
                "parsed_ts_idx",
                "2020-08-01 00:00:10.010",
                "2020-09-01 00:02:10.076",
                {
                    "ts_idx": {
                        "ts_col": "event_ts",
                        "series_ids": ["symbol"],
                        "ts_fmt": "yyyy-MM-dd HH:mm:ss.SSS",
                    },
                    "tsdf_constructor": "fromStringTimestamp",
                    "df": {
                        "schema": "symbol string, event_ts string, trade_pr float",
                        "data": [
                            ["S1", "2020-08-01 00:00:10.010", 349.21],
                            ["S1", "2020-08-01 00:01:12.021", 351.32],
                            ["S1", "2020-09-01 00:02:10.032", 361.1],
                            ["S2", "2020-08-01 00:01:10.054", 743.01],
                            ["S2", "2020-08-01 00:01:24.065", 751.92],
                            ["S2", "2020-09-01 00:02:10.076", 761.10],
                        ],
                    },
                },
            ),
            (
                "parsed_date_idx",
                "2020-08-01",
                "2020-08-03",
                {
                    "ts_idx": {
                        "ts_col": "date",
                        "series_ids": ["station"],
                        "ts_fmt": "yyyy-MM-dd",
                    },
                    "tsdf_constructor": "fromStringTimestamp",
                    "df": {
                        "schema": "station string, date string, temp float",
                        "data": [
                            ["LGA", "2020-08-01", 27.58],
                            ["LGA", "2020-08-02", 28.79],
                            ["LGA", "2020-08-03", 28.53],
                            ["YYZ", "2020-08-01", 24.16],
                            ["YYZ", "2020-08-02", 22.25],
                            ["YYZ", "2020-08-03", 20.62],
                        ],
                    },
                },
            ),
        ]
    )
    def test_between_inclusive(
        self, init_tsdf_id, start_ts, end_ts, expected_tsdf_dict
    ):
        # load TSDF
        tsdf = self.get_test_data(init_tsdf_id).as_tsdf()
        # slice at timestamp
        between_tsdf = tsdf.between(start_ts, end_ts, inclusive=True)
        # validate the slice
        expected_tsdf = TestDataFrame(self.spark, expected_tsdf_dict).as_tsdf()
        self.assertDataFrameEquality(between_tsdf, expected_tsdf)

    @parameterized.expand(
        [
            (
                "simple_ts_idx",
                2,
                {
                    "ts_idx": {"ts_col": "event_ts", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, event_ts string, trade_pr float",
                        "ts_convert": ["event_ts"],
                        "data": [
                            ["S1", "2020-08-01 00:00:10", 349.21],
                            ["S1", "2020-08-01 00:01:12", 351.32],
                            ["S2", "2020-08-01 00:01:10", 743.01],
                            ["S2", "2020-08-01 00:01:24", 751.92],
                        ],
                    },
                },
            ),
            (
                "simple_ts_no_series",
                2,
                {
                    "ts_idx": {"ts_col": "event_ts", "series_ids": []},
                    "df": {
                        "schema": "event_ts string, trade_pr float",
                        "ts_convert": ["event_ts"],
                        "data": [
                            ["2020-08-01 00:00:10", 349.21],
                            ["2020-08-01 00:01:10", 743.01],
                        ],
                    },
                },
            ),
            (
                "simple_date_idx",
                2,
                {
                    "ts_idx": {"ts_col": "date", "series_ids": ["station"]},
                    "df": {
                        "schema": "station string, date string, temp float",
                        "date_convert": ["date"],
                        "data": [
                            ["LGA", "2020-08-01", 27.58],
                            ["LGA", "2020-08-02", 28.79],
                            ["YYZ", "2020-08-01", 24.16],
                            ["YYZ", "2020-08-02", 22.25],
                        ],
                    },
                },
            ),
            (
                "ordinal_double_index",
                2,
                {
                    "ts_idx": {"ts_col": "event_ts_dbl", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, event_ts_dbl double, trade_pr float",
                        "data": [
                            ["S1", 0.13, 349.21],
                            ["S1", 1.207, 351.32],
                            ["S2", 0.005, 743.01],
                            ["S2", 0.1, 751.92],
                        ],
                    },
                },
            ),
            (
                "ordinal_int_index",
                2,
                {
                    "ts_idx": {"ts_col": "order", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, order int, trade_pr float",
                        "data": [
                            ["S1", 1, 349.21],
                            ["S1", 20, 351.32],
                            ["S2", 0, 743.01],
                            ["S2", 1, 751.92],
                        ],
                    },
                },
            ),
            (
                "parsed_ts_idx",
                2,
                {
                    "ts_idx": {
                        "ts_col": "event_ts",
                        "series_ids": ["symbol"],
                        "ts_fmt": "yyyy-MM-dd HH:mm:ss.SSS",
                    },
                    "tsdf_constructor": "fromStringTimestamp",
                    "df": {
                        "schema": "symbol string, event_ts string, trade_pr float",
                        "data": [
                            ["S1", "2020-08-01 00:00:10.010", 349.21],
                            ["S1", "2020-08-01 00:01:12.021", 351.32],
                            ["S2", "2020-08-01 00:01:10.054", 743.01],
                            ["S2", "2020-08-01 00:01:24.065", 751.92],
                        ],
                    },
                },
            ),
            (
                "parsed_date_idx",
                2,
                {
                    "ts_idx": {
                        "ts_col": "date",
                        "series_ids": ["station"],
                        "ts_fmt": "yyyy-MM-dd",
                    },
                    "tsdf_constructor": "fromStringTimestamp",
                    "df": {
                        "schema": "station string, date string, temp float",
                        "data": [
                            ["LGA", "2020-08-01", 27.58],
                            ["LGA", "2020-08-02", 28.79],
                            ["YYZ", "2020-08-01", 24.16],
                            ["YYZ", "2020-08-02", 22.25],
                        ],
                    },
                },
            ),
        ]
    )
    def test_earliest(self, init_tsdf_id, num_records, expected_tsdf_dict):
        # load TSDF
        tsdf = self.get_test_data(init_tsdf_id).as_tsdf()
        # get earliest timestamp
        earliest_ts = tsdf.earliest(n=num_records)
        # validate the timestamp
        expected_tsdf = TestDataFrame(self.spark, expected_tsdf_dict).as_tsdf()
        self.assertDataFrameEquality(earliest_ts, expected_tsdf)

    @parameterized.expand(
        [
            (
                "simple_ts_idx",
                2,
                {
                    "ts_idx": {"ts_col": "event_ts", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, event_ts string, trade_pr float",
                        "ts_convert": ["event_ts"],
                        "data": [
                            ["S1", "2020-09-01 00:19:12", 362.1],
                            ["S1", "2020-09-01 00:02:10", 361.1],
                            ["S2", "2020-09-01 00:20:42", 762.33],
                            ["S2", "2020-09-01 00:02:10", 761.10],
                        ],
                    },
                },
            ),
            (
                "simple_ts_no_series",
                4,
                {
                    "ts_idx": {"ts_col": "event_ts", "series_ids": []},
                    "df": {
                        "schema": "event_ts string, trade_pr float",
                        "ts_convert": ["event_ts"],
                        "data": [
                            ["2020-09-01 00:20:42", 762.33],
                            ["2020-09-01 00:19:12", 362.1],
                            ["2020-09-01 00:02:10", 361.1],
                            ["2020-08-01 00:01:24", 751.92],
                        ],
                    },
                },
            ),
            (
                "simple_date_idx",
                3,
                {
                    "ts_idx": {"ts_col": "date", "series_ids": ["station"]},
                    "df": {
                        "schema": "station string, date string, temp float",
                        "date_convert": ["date"],
                        "data": [
                            ["LGA", "2020-08-04", 25.57],
                            ["LGA", "2020-08-03", 28.53],
                            ["LGA", "2020-08-02", 28.79],
                            ["YYZ", "2020-08-04", 20.65],
                            ["YYZ", "2020-08-03", 20.62],
                            ["YYZ", "2020-08-02", 22.25],
                        ],
                    },
                },
            ),
            (
                "ordinal_double_index",
                1,
                {
                    "ts_idx": {"ts_col": "event_ts_dbl", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, event_ts_dbl double, trade_pr float",
                        "data": [["S1", 24.357, 362.1], ["S2", 10.0, 762.33]],
                    },
                },
            ),
            (
                "ordinal_int_index",
                3,
                {
                    "ts_idx": {"ts_col": "order", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, order int, trade_pr float",
                        "data": [
                            ["S1", 243, 362.1],
                            ["S1", 127, 361.1],
                            ["S1", 20, 351.32],
                            ["S2", 100, 762.33],
                            ["S2", 10, 761.10],
                            ["S2", 1, 751.92],
                        ],
                    },
                },
            ),
            (
                "parsed_ts_idx",
                3,
                {
                    "ts_idx": {
                        "ts_col": "event_ts",
                        "series_ids": ["symbol"],
                        "ts_fmt": "yyyy-MM-dd HH:mm:ss.SSS",
                    },
                    "tsdf_constructor": "fromStringTimestamp",
                    "df": {
                        "schema": "symbol string, event_ts string, trade_pr float",
                        "data": [
                            ["S1", "2020-09-01 00:19:12.043", 362.1],
                            ["S1", "2020-09-01 00:02:10.032", 361.1],
                            ["S1", "2020-08-01 00:01:12.021", 351.32],
                            ["S2", "2020-09-01 00:20:42.087", 762.33],
                            ["S2", "2020-09-01 00:02:10.076", 761.10],
                            ["S2", "2020-08-01 00:01:24.065", 751.92],
                        ],
                    },
                },
            ),
            (
                "parsed_date_idx",
                1,
                {
                    "ts_idx": {
                        "ts_col": "date",
                        "series_ids": ["station"],
                        "ts_fmt": "yyyy-MM-dd",
                    },
                    "tsdf_constructor": "fromStringTimestamp",
                    "df": {
                        "schema": "station string, date string, temp float",
                        "data": [
                            ["LGA", "2020-08-04", 25.57],
                            ["YYZ", "2020-08-04", 20.65],
                        ],
                    },
                },
            ),
        ]
    )
    def test_latest(self, init_tsdf_id, num_records, expected_tsdf_dict):
        # load TSDF
        tsdf = self.get_test_data(init_tsdf_id).as_tsdf()
        # get earliest timestamp
        latest_ts = tsdf.latest(n=num_records)
        # validate the timestamp
        expected_tsdf = TestDataFrame(self.spark, expected_tsdf_dict).as_tsdf()
        self.assertDataFrameEquality(latest_ts, expected_tsdf)

    @parameterized.expand(
        [
            (
                "simple_ts_idx",
                "2020-09-01 00:02:10",
                2,
                {
                    "ts_idx": {"ts_col": "event_ts", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, event_ts string, trade_pr float",
                        "ts_convert": ["event_ts"],
                        "data": [
                            ["S1", "2020-09-01 00:02:10", 361.1],
                            ["S1", "2020-08-01 00:01:12", 351.32],
                            ["S2", "2020-09-01 00:02:10", 761.10],
                            ["S2", "2020-08-01 00:01:24", 751.92],
                        ],
                    },
                },
            ),
            (
                "simple_ts_no_series",
                "2020-09-01 00:19:12",
                3,
                {
                    "ts_idx": {"ts_col": "event_ts", "series_ids": []},
                    "df": {
                        "schema": "event_ts string, trade_pr float",
                        "ts_convert": ["event_ts"],
                        "data": [
                            ["2020-09-01 00:19:12", 362.1],
                            ["2020-09-01 00:02:10", 361.1],
                            ["2020-08-01 00:01:24", 751.92],
                        ],
                    },
                },
            ),
            (
                "simple_date_idx",
                "2020-08-03",
                2,
                {
                    "ts_idx": {"ts_col": "date", "series_ids": ["station"]},
                    "df": {
                        "schema": "station string, date string, temp float",
                        "date_convert": ["date"],
                        "data": [
                            ["LGA", "2020-08-03", 28.53],
                            ["LGA", "2020-08-02", 28.79],
                            ["YYZ", "2020-08-03", 20.62],
                            ["YYZ", "2020-08-02", 22.25],
                        ],
                    },
                },
            ),
            (
                "ordinal_double_index",
                10.0,
                4,
                {
                    "ts_idx": {"ts_col": "event_ts_dbl", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, event_ts_dbl double, trade_pr float",
                        "data": [
                            ["S1", 10.0, 361.1],
                            ["S1", 1.207, 351.32],
                            ["S1", 0.13, 349.21],
                            ["S2", 10.0, 762.33],
                            ["S2", 1.0, 761.10],
                            ["S2", 0.1, 751.92],
                            ["S2", 0.005, 743.01],
                        ],
                    },
                },
            ),
            (
                "ordinal_int_index",
                1,
                1,
                {
                    "ts_idx": {"ts_col": "order", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, order int, trade_pr float",
                        "data": [["S1", 1, 349.21], ["S2", 1, 751.92]],
                    },
                },
            ),
            (
                "parsed_ts_idx",
                "2020-09-01 00:02:10.000",
                2,
                {
                    "ts_idx": {
                        "ts_col": "event_ts",
                        "series_ids": ["symbol"],
                        "ts_fmt": "yyyy-MM-dd HH:mm:ss.SSS",
                    },
                    "tsdf_constructor": "fromStringTimestamp",
                    "df": {
                        "schema": "symbol string, event_ts string, trade_pr float",
                        "data": [
                            ["S1", "2020-08-01 00:01:12.021", 351.32],
                            ["S1", "2020-08-01 00:00:10.010", 349.21],
                            ["S2", "2020-08-01 00:01:24.065", 751.92],
                            ["S2", "2020-08-01 00:01:10.054", 743.01],
                        ],
                    },
                },
            ),
            (
                "parsed_date_idx",
                "2020-08-03",
                3,
                {
                    "ts_idx": {
                        "ts_col": "date",
                        "series_ids": ["station"],
                        "ts_fmt": "yyyy-MM-dd",
                    },
                    "tsdf_constructor": "fromStringTimestamp",
                    "df": {
                        "schema": "station string, date string, temp float",
                        "data": [
                            ["LGA", "2020-08-03", 28.53],
                            ["LGA", "2020-08-02", 28.79],
                            ["LGA", "2020-08-01", 27.58],
                            ["YYZ", "2020-08-03", 20.62],
                            ["YYZ", "2020-08-02", 22.25],
                            ["YYZ", "2020-08-01", 24.16],
                        ],
                    },
                },
            ),
        ]
    )
    def test_priorTo(self, init_tsdf_id, ts, n, expected_tsdf_dict):
        # load TSDF
        tsdf = self.get_test_data(init_tsdf_id).as_tsdf()
        # slice at timestamp
        prior_tsdf = tsdf.priorTo(ts, n=n)
        # validate the slice
        expected_tsdf = TestDataFrame(self.spark, expected_tsdf_dict).as_tsdf()
        self.assertDataFrameEquality(prior_tsdf, expected_tsdf)

    @parameterized.expand(
        [
            (
                "simple_ts_idx",
                "2020-09-01 00:02:10",
                1,
                {
                    "ts_idx": {"ts_col": "event_ts", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, event_ts string, trade_pr float",
                        "ts_convert": ["event_ts"],
                        "data": [
                            ["S1", "2020-09-01 00:02:10", 361.1],
                            ["S2", "2020-09-01 00:02:10", 761.10],
                        ],
                    },
                },
            ),
            (
                "simple_ts_no_series",
                "2020-08-01 00:01:24",
                3,
                {
                    "ts_idx": {"ts_col": "event_ts", "series_ids": []},
                    "df": {
                        "schema": "event_ts string, trade_pr float",
                        "ts_convert": ["event_ts"],
                        "data": [
                            ["2020-08-01 00:01:24", 751.92],
                            ["2020-09-01 00:02:10", 361.1],
                            ["2020-09-01 00:19:12", 362.1],
                        ],
                    },
                },
            ),
            (
                "simple_date_idx",
                "2020-08-02",
                5,
                {
                    "ts_idx": {"ts_col": "date", "series_ids": ["station"]},
                    "df": {
                        "schema": "station string, date string, temp float",
                        "date_convert": ["date"],
                        "data": [
                            ["LGA", "2020-08-02", 28.79],
                            ["LGA", "2020-08-03", 28.53],
                            ["LGA", "2020-08-04", 25.57],
                            ["YYZ", "2020-08-02", 22.25],
                            ["YYZ", "2020-08-03", 20.62],
                            ["YYZ", "2020-08-04", 20.65],
                        ],
                    },
                },
            ),
            (
                "ordinal_double_index",
                10.0,
                2,
                {
                    "ts_idx": {"ts_col": "event_ts_dbl", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, event_ts_dbl double, trade_pr float",
                        "data": [
                            ["S1", 10.0, 361.1],
                            ["S1", 24.357, 362.1],
                            ["S2", 10.0, 762.33],
                        ],
                    },
                },
            ),
            (
                "ordinal_int_index",
                10,
                2,
                {
                    "ts_idx": {"ts_col": "order", "series_ids": ["symbol"]},
                    "df": {
                        "schema": "symbol string, order int, trade_pr float",
                        "data": [
                            ["S1", 20, 351.32],
                            ["S1", 127, 361.1],
                            ["S2", 10, 761.10],
                            ["S2", 100, 762.33],
                        ],
                    },
                },
            ),
            (
                "parsed_ts_idx",
                "2020-09-01 00:02:10",
                3,
                {
                    "ts_idx": {
                        "ts_col": "event_ts",
                        "series_ids": ["symbol"],
                        "ts_fmt": "yyyy-MM-dd HH:mm:ss.SSS",
                    },
                    "tsdf_constructor": "fromStringTimestamp",
                    "df": {
                        "schema": "symbol string, event_ts string, trade_pr float",
                        "data": [
                            ["S1", "2020-09-01 00:02:10.032", 361.1],
                            ["S1", "2020-09-01 00:19:12.043", 362.1],
                            ["S2", "2020-09-01 00:02:10.076", 761.10],
                            ["S2", "2020-09-01 00:20:42.087", 762.33],
                        ],
                    },
                },
            ),
            (
                "parsed_date_idx",
                "2020-08-03",
                2,
                {
                    "ts_idx": {
                        "ts_col": "date",
                        "series_ids": ["station"],
                        "ts_fmt": "yyyy-MM-dd",
                    },
                    "tsdf_constructor": "fromStringTimestamp",
                    "df": {
                        "schema": "station string, date string, temp float",
                        "data": [
                            ["LGA", "2020-08-03", 28.53],
                            ["LGA", "2020-08-04", 25.57],
                            ["YYZ", "2020-08-03", 20.62],
                            ["YYZ", "2020-08-04", 20.65],
                        ],
                    },
                },
            ),
        ]
    )
    def test_subsequentTo(self, init_tsdf_id, ts, n, expected_tsdf_dict):
        # load TSDF
        tsdf = self.get_test_data(init_tsdf_id).as_tsdf()
        # slice at timestamp
        subseq_tsdf = tsdf.subsequentTo(ts, n=n)
        # validate the slice
        expected_tsdf = TestDataFrame(self.spark, expected_tsdf_dict).as_tsdf()
        self.assertDataFrameEquality(subseq_tsdf, expected_tsdf)
