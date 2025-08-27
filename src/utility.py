from collections.abc import Iterable
from collections import OrderedDict
from copy import deepcopy
import datasets
import datetime
from dateparser import parse
from functools import cache
import numpy as np
import os
import pandas as pd
from pandas import DataFrame
from queue import Queue
from re import compile
import signal
import sys
from threading import Thread
from tldextract import extract
from typing import Union, Callable, Literal, Optional
import wikidataintegrator as wdi

# The max_retries parameter doesn't actually seem to work; but if you
# set it to zero, the query doesn't run.
execute_sparql_query = (
    lambda query: wdi.wdi_core.WDItemEngine.execute_sparql_query(
        query, user_agent=WIKIDATA_USER_AGENT, max_retries=1, as_dataframe=True
    )
)
execute_sparql_query.__doc__ = (
    wdi.wdi_core.WDItemEngine.execute_sparql_query.__doc__
)

### Global variables
# Create an agnostic type over Pandas Dataframes and HuggingFace datasets.
type DatasetObject = Union[DataFrame, datasets.Dataset]

__DEBUG_MODE = False

HF_HUB_TOKEN = None
RUN_ALL = False  # Run maximal tasks such as extracting date ranges
# As above, but for expensive-yet-fruitless tasks
RUN_FALSE_ENDS = False
# PERFORM_TESTS = False
WIKIDATA_USER_AGENT = "Scraping for thesis about source reliability prediction at the University of Copenhagen. [redacted]"
ALL_DATASETS = [
    "AVeriTeC",
    "Domain Ratings",
    "DRUID+",
    "Emergent.Info",
    "FakeHealth",
    "FakeNewsCorpus",
    "FakeNewsNet",
    "GossipCop",
    "Liar-PLUS",
    "MisInfo General",
    "MM-COVID19",
    "MultiFC",
    "News Media Reliability EMNLP",
    "PolitiFact Fact Check",
    "ReCoVery",
]

# # Filenames
SOURCE_ID_FILENAME = "source_id.csv"
SOURCE_ID_LOG_FILENAME = "_source_id_log.txt"
NEIGHBOR_DATASETS_DIR = "neighbor_datasets"
DATA_DIR = "data"

### Classes
class QueryFailureWatcher:
    """
    This context manager serves two purposes.

    First, the builtin parameters for stopping the WDI query-executor from retrying do not work. This context manager tries in less elegant fashion to raise a RuntimeError if the executor starts backing off (typically as a result of too large queries/queries that time out). In doing so, it also fulfills its second purpose in handling saving and loading in cases of exceptions, including KeyboardInterrupt.
    """

    def __init__(
        self,
        queue: Queue,
        worker: Thread,
        log_queue: Optional[Queue] = None,
        logger: Optional[Thread] = None,
        behavior: Union[Literal["raise"], Literal["skip"]] = "raise",
    ):
        assert behavior in ["raise", "skip"]
        self.queue = queue
        self.worker = worker
        self.log_queue = log_queue
        self.logger = logger
        self.behavior = behavior
        self.__termination_signal = -1

    def __enter__(self):
        signal.signal(signal.SIGTERM, self.__handle_signal)
        self.original_write = sys.stdout.write
        def watched_write(text):
            if "Backing off " in text:
                if self.behavior == "raise":
                    raise RuntimeError("Query handler backed off! With the text:", text)
                else:
                    self.log_queue.put(
                        f"Query handler backed off! With the text: {text}\n\n"
                    )
            else:
                return self.original_write(text)
        sys.stdout.write = watched_write
        return None

    def __exit__(self, exception_class, exception_value, traceback):
        sys.stdout.write = self.original_write
        if exception_class:
            self.queue.put(self.__termination_signal)
            self.worker.join()
            if self.log_queue:
                self.log_queue.put(self.__termination_signal)
            if self.logger:
                self.logger.join()
            exception_value.__traceback__ = traceback
            raise exception_value
        return True

    def __handle_signal(self, signum, frame):
        if signum == signal.SIGINT:
            exc = KeyboardInterrupt()
        else:  # Handle termination
            exc = SystemExit()
        self.__exit__(type(exc), exc, frame)
        if signum == signal.SIGTERM:
            sys.exit(0)


### Functions
@cache
def _cached_parse(date: str):
    return parse(date, settings={"DEFAULT_LANGUAGES": ["en"]})


def _collapse_nested_str_list_to_iter(nested_str_list) -> iter:
    if nested_str_list is not None and len(nested_str_list):
        for l in nested_str_list:
            if not (isinstance(l, str)):
                yield from _collapse_nested_str_list_to_iter(l)
            else:
                yield l

########## TEMP


def populate_nodes_df_from_sro_df(sro_df: DataFrame, nodes_df):
    """
    Given a properly formatted dataframe of subject-relation-object triples and a node accumulation dataframe, add to the node accumulation dataframe the new items among the subjects and objects of the SRO-dataframe.

    Parameters
    -
    sro_df: pd.DataFrame
        A dataframe with subject, relation and object columns
    nodes_df: pd.DataFrame
        A dataframe of nodes
    """
    cols_to_keep = nodes_df.columns
    subject_nodes_df = aggregate_df_class_labels(sro_df, target="subject")
    object_nodes_df = aggregate_df_class_labels(sro_df, target="object")
    new_subjects = rename_triple_roles_to_source(
        subject_nodes_df[~subject_nodes_df["subject"].isin(nodes_df["source"])],
        "subject",
    )
    new_objects = rename_triple_roles_to_source(
        object_nodes_df[
            ~object_nodes_df["object"].isin(
                pd.concat(
                    [nodes_df["source"], new_subjects["source"]]
                ).drop_duplicates()
            )
        ],
        "object",
    )
    return pd.concat([nodes_df, new_subjects, new_objects])[cols_to_keep]


def rename_triple_roles_to_source(
    df, role: Union[Literal["subject"], Literal["object"]]
):
    """
    For a dataframe with columns with a "subject" or "object" prefix, replace it with "source".
    """
    return df.rename(columns={col: col.replace(role, "source") for col in df.columns})


RANKED_ACCEPTED_SOURCE_CLASSES = OrderedDict(
    {
        (
            "misinformation creator",
            "politician",
            "business magnate",
            "celebrity",
            "Content creator",  # Capital C appears in dataset??
            "content creator",
            "leader",
            "manager",
            "economist",
            "jurist",
            "knowledge worker",
            "scholar",
            "functionary",
            "official",
            "media professional",
            "activist",
            "non-fiction writer",
            "researcher",
        ): ("person", 1),
        (
            "podcast show",
            "company",
            "business and professional associations, unions",
            "advocacy group",
            "broadcaster",
            "committee",
            "council",
            "educational organization",
            "GLAM",
            "media company",
            "medical organization",
            "nonprofit organization",
            "political institution",
            "think tank",
            "university",
            "organization",
            "None",  # Default here
            "special interest group",
        ): ("organization", 2),
        ("publication", "non-fiction", "website", "periodical"): ("data", 3),
    }
)

CLASS_CATEGORY_RANK_MAP = OrderedDict(
    {
        class_: pair
        for classes, pair in RANKED_ACCEPTED_SOURCE_CLASSES.items()
        for class_ in (classes if isinstance(classes, tuple) else (classes,))
    }
)


def _add_category_column(row, target="source"):
    """
    Given a dataframe with the target ClassLabel column, add a corresponding column of the highest ranked category that the set of labels belongs to.
    """
    if None not in row[f"{target}ClassLabels"] and "nan" not in row[
        f"{target}ClassLabels"
    ].astype("str"):
        if "Content creator" in row[f"{target}ClassLabels"]:
            breakpoint
        row[f"{target}Category"] = min(
            (CLASS_CATEGORY_RANK_MAP[label] for label in row[f"{target}ClassLabels"]),
            key=lambda category_rank_pair: category_rank_pair[1],
        )[0]
    else:
        row[f"{target}Category"] = None
    return row


def aggregate_class_labels(df, target="source") -> pd.DataFrame:
    """
    Take an (exploded) dataframe of multiple labels per source and convert it to a dataframe where the labels are aggregated in arrays.
    """
    if f"{target}ClassLabel" not in df.columns:
        print(f"Did not find {target}ClassLabel! Found {df.columns}.")
        raise ValueError
    df = df.dropna(subset=f"{target}ClassLabel")
    result = df.groupby(target).first()
    result[f"{target}ClassLabels"] = df.groupby(target)[f"{target}ClassLabel"].unique()
    return result.reset_index()


def aggregate_df_class_labels(df, target="source", **kwargs) -> pd.DataFrame:
    """
    Take an exploded dataframe with multiple labels per source and compress it such that each source is associated with a list of labels, further add a column for the highest ranked category the source belongs to.
    """
    df = aggregate_class_labels(df, target=target)
    df = df.apply(
        lambda row: _add_category_column(row, target=target, **kwargs), axis=1
    )
    return df


########## TEMP

def add_binary_label_column(dataset, label: bool):
    return dataset.add_column("label", [label]*len(dataset))


def extract_dataset_with_authors_from_list_column(dataset, column_name):
    """
    Given a dataset that has explicitly or implicitly array-formatted authors in a column, return the dataset where the author column is replaced to contain the exploded rows for each participating author in each array.
    """
    col_dtype = dataset.features[column_name].dtype
    assert col_dtype in [
        "string",
        "list",
    ], f"Unhandled author column dtype: {col_dtype}"
    rows_out = []
    if col_dtype == "string":
        author_name = compile("'([^']+[^O])'")
        for i, array_str in enumerate(dataset[column_name]):
            if array_str is not None:
                if match := author_name.findall(array_str):
                    for name in match:
                        row_out = deepcopy(dataset[i])
                        row_out[column_name] = name
                        rows_out.append(row_out)
    elif col_dtype == "list":
        for i, arr in enumerate(dataset[column_name]):
            if arr is not None and len(arr):
                for name in arr:
                    row_out = deepcopy(dataset[i])
                    row_out[column_name] = name
                    rows_out.append(row_out)
    dataset_out = datasets.Dataset.from_list(rows_out)
    return dataset_out


def extract_set_of_authors_from_list_column(dataset, column_name):
    """
    Extract set of authors contained in a list column, whether truly list or list-like string. List-like string is only superficially cleaned.
    """
    col_dtype = dataset.features[column_name].dtype
    assert col_dtype in [
        "string",
        "list",
    ], f"Unhandled author column dtype: {col_dtype}"
    authors = set()
    if col_dtype == "string":
        # Pattern of names enclosed in ' but not ending e.g. in O'Donald
        author_name = compile("'([^']+[^O])'")

        for array_str in dataset[column_name]:
            if array_str is not None:
                if match := author_name.findall(array_str):
                    for name in match:
                        authors.add(name)
    elif col_dtype == "list":
        for arr in dataset[column_name]:
            if arr is not None and len(arr):
                authors.update(arr)
    return authors


def extract_set_of_domains_from_urls(urls: Iterable):
    domains_from_urls = set()
    for url in list(urls):
        if url is not None:
            extracted_domain_info = extract(url)
            """
            There is some slightly ugly logic here. Some subdomains are empty. Other subdomains are "www". Other subdomains yet are "www.*". I strip any "www." prefix, expecting this to line up with how other domains are normalized.
            """
            # This error mode has been observed. In the observed case
            # (www.gov.uk), we could return the suffix instead, but rather
            # just drop it for generalizability.
            if extracted_domain_info.domain == "www":
                continue

            if (
                not (extracted_domain_info.subdomain)
                or extracted_domain_info.subdomain == "www"
            ):
                domain_string = (
                    extracted_domain_info.domain
                    + "."
                    + extracted_domain_info.suffix
                )
            else:
                domain_string = (
                    extracted_domain_info.subdomain.strip("www").strip(".")
                    + "."
                    + extracted_domain_info.domain
                    + "."
                    + extracted_domain_info.suffix
                )

            # Keeping this assertion for debugging is useful given the ad hoc
            # nature of this function.
            assert ("www." not in domain_string) and not (
                domain_string.startswith(".")
            ), {
                "url": url,
                "string": domain_string,
                "subdomain": extracted_domain_info.subdomain,
                "domain": extracted_domain_info.domain,
                "suffix": extracted_domain_info.suffix,
            }

            domains_from_urls.add(domain_string)
    return domains_from_urls


def extract_domain_from_url_in_row(row, col_name):
    url = row[col_name]
    if url is not None:
        extracted_domain_info = extract(url)
        match (extracted_domain_info.domain, extracted_domain_info.suffix):
            case (domain, suffix):
                row[col_name] = domain + "." + suffix
            case (domain, None):
                row[col_name] = domain
    return row


def in_data_dir(path):
    return os.path.join(DATA_DIR, path)


def in_neighbors_dir(path):
    return os.path.join(NEIGHBOR_DATASETS_DIR, path)


def pdw(f: Callable, dataset: datasets.Dataset, **kwargs):
    """Convenience function for performing a function inside of a Pandas 'wrapper'"""
    return datasets.Dataset.from_pandas(f(dataset.to_pandas(), **kwargs))


def print_df_summary_statistics(df):
    print("Size:")
    print(len(df))
    print("Proportion of relationLabels:")
    print(df.value_counts(subset="relationLabel", normalize=True), end="\n\n")
    print("Most common items, absolute value:")
    print(
        pd.concat(
            [
                df["subjectLabel"],
                df["objectLabel"],
            ]
        ).value_counts()[:20],
        end="\n\n",
    )
    print("Most common items, normalized")
    print(
        pd.concat(
            [
                df["subjectLabel"],
                df["objectLabel"],
            ]
        ).value_counts(
            normalize=True
        )[:20],
        end="\n\n",
    )
    print("Most common subject-relation pairs:")
    print(df.value_counts(subset=["subjectLabel", "relationLabel"])[:20], end="\n\n")

    print("Most common relation-object pairs:")
    print(df.value_counts(subset=["relationLabel", "objectLabel"])[:20], end="\n\n")

    for rel in df["relationLabel"].unique():
        print(f"Subjects of {rel}:")
        print(
            df[df["relationLabel"] == rel].value_counts(subset=["subjectLabel"])[:3],
            end="\n\n",
        )

        print(f"Objects of {rel}:")
        print(
            df[df["relationLabel"] == rel].value_counts(subset=["objectLabel"])[:3],
            end="\n\n",
        )

    print(f"Mean number of relations:")
    print(
        (
            df.groupby(
                [
                    ["subjectLabel", "relationLabel", "objectLabel"]
                ]
            ).count()
            / df["subjectLabel"].nunique()
        ),
        sep="\n\n",
    )
    print(f"Median number of relations:")
    print(
        df.groupby(["subjectLabel", "relationLabel"])
        .size()
        .groupby("relationLabel")
        .median()
    )


def recursive_typecheck(obj) -> str:
    if isinstance(obj, Iterable) and not isinstance(obj, str) and len(obj) > 0:
        return (
            str(type(obj)) + " [" + recursive_typecheck(next(iter(obj))) + "]"
        )
    else:
        return str(type(obj))


# Not terribly fast.
def retrieve_heterogeneously_formatted_years(dates: Iterable):
    years = set()
    for date in dates:
        # We need to check for two failure modes: Dates may be None, and
        # attempted parses of dates may be None.
        if date is not None:
            try:
                years.add(_cached_parse(date).strftime("%Y"))
            except Exception:
                continue
    return sorted(years)


def retrieve_unique_nested_strings(nested_str_list) -> set:
    return set(_collapse_nested_str_list_to_iter(nested_str_list))


def return_timestring():
    return datetime.datetime.strftime(
        datetime.datetime.now(), "%a. the %d. at %H.%M.%S."
    )


def str_list_to_lowercase(str_list):
    # Also drops NoneValues and trailing whitespace
    return list(
        map(
            lambda s: s.lower().strip(),
            filter(lambda s: s is not None, str_list),
        )
    )


def stringsafe(s):
    return s.replace('"', '\\"').replace("'", "\\'")


def wikidata_search(query, **kwargs):
    return wdi.wdi_core.WDItemEngine.get_wd_search_results(
        query,
        dict_id_all_info=True,
        user_agent=WIKIDATA_USER_AGENT,
        **kwargs,
    )
# Inherit the docstring
wikidata_search.__doc__ = wdi.wdi_core.WDItemEngine.get_wd_search_results.__doc__