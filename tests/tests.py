if __name__ == "__main__":
   import os
   os.chdir("..")
from src.utility import *

import unittest as ut
import datasets
import pandas as pd

class TestSparqlGenerator(ut.TestCase):
    """
    Note that this testcase only tests for queries not being malformed, it does not check that they have any hits.
    """
    
    @classmethod
    def setUpClass(cls):
        cls.g = SparqlGenerator()
        cls.properties = {"instance of": "P31", "born at": "P569"}
        cls.items = {"human": "Q5", "patent": "Q253623"}
        df = pd.read_csv(os.path.join(DATA_DIR, "wikifake.csv")).rename(
            columns={
                "wikidataID": "source",
                "source_name_from_nl": "source_name",
            }
        ).fillna("").astype("string")
        df_mini = pd.DataFrame(
            {
                "source": ["Q22686", "Q76"],
                "source_name": ["Donald J. Trump", "Barack Obama"],
                "website": [
                    "http://donaldjtrump.com/",
                    "http://barackobama.com/",
                ],
            }
        )
        cls.df = df
        cls.df_mini = df_mini
        cls.dataset = datasets.Dataset.from_pandas(df)
        cls.dataset_mini = datasets.Dataset.from_pandas(df_mini)

    def setUp(self):
        self.maxDiff = None


    def test_validation(self):
        self.assertRaises(
            AssertionError,
            self.g.validate,
            pd.DataFrame(
                {
                    "non canonical col1": [1, 2],
                    "non canonical col2": ["a", "b"],
                }
            ),
        )

    def test_agnosticism(self):
        self.assertEqual(
            self.g.for_one_hop_property(
                self.df_mini,
                self.items,
                self.properties,
                bidirectional=False,
                filter_humans=False,
                filter_patents=False,
            ),
            self.g.for_one_hop_property(
                self.dataset_mini,
                self.items,
                self.properties,
                bidirectional=False,
                filter_humans=False,
                filter_patents=False,
            ),
        )

    def test_for_one_hop_property(self):
        execute_sparql_query(
            self.g.for_one_hop_property(
                self.df_mini,
                self.items,
                self.properties,
                bidirectional=False,
                filter_humans=False,
                filter_patents=False,
            )
        )

    def test_for_owners_of_official_websites(self):
        execute_sparql_query(
            self.g.for_owners_of_official_websites(
                self.df_mini, uri_is_inferred=True
            )
        )
        # Not currently maintained
        # self.__assert_successful_query(
        #     self.g.for_owners_of_official_websites(
        #         self.df_mini,
        #         uri_is_inferred=False
        #     )
        # )

    def test_for_valid_sources(self):
        execute_sparql_query(
            self.g.for_valid_sources(
                self.df_mini,
                self.properties,
                self.items,
            )
        )