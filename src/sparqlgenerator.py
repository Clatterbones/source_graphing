"""
Library for generating SPARQL queries.

Any dataset passed must follow the canonical naming scheme. This means that its columns must have names that contain a subset of:
    Externally supplied columns:
        "source_name": a string used as a name (optionally, an identifier) in the input dataset
        "website": A column containing URIs that serve as website values on WikiData

    Internally supplied columns:
        "source": WikiData sourceIDs
        "source_label": WikiData labels
        "source_description": WikiData descriptions

        "subject" : WikiData subject IDs of triples
        "subjectLabel": WikiData subject labels
        "subjectDescription": : WikiData subject descriptions

        "relation": WikiData IDs of relations of triples
        "relationLabel": labels of relations of triples
        "relationDescription": descriptions of relatinos of triples

        "object": WikiData object IDs of triples
        "objectLabel": labels of objects of triples
        "objectDescription: descriptions of objects of triples"

That is, most of these column names are generated and used internally, but can occur in existing datasets, especially when generating multiple queries in series. The "source" name is typically used when single items are the center of a query, whereas "subject" and "object" are used when relation triples are the center of the query.
"""

if __name__ == "__main__":
    import os

    os.chdir("..")
from src.utility import *

import pandas as pd
from typing import Iterable, Literal, Union


__CANONICAL_FEATURES = {
    "source_name",
    "website",
    "source",
    "sourceLabel",
    "sourceDescription",
    "sourceClassLabel",
    "sourceCategory",
    "relation",
    "relationLabel",
    "relationDescription",
    "object",
    "objectLabel",
    "objectDescription",
    "objectClassLabel",
    "objectCategory",
}

__IMPORTANT_STRINGS = {
    "SUBCLASS OF": "wdt:P279*",
    "CLASS OF FICTIONAL ENTITIES": "wd:Q15831596",
    "OFFICIAL WEBSITE OF": "^wdt:P856",
    "ITEM": "wd:",
    "PROPERTY": "wdt:",
    "BORN AT": "wdt:P569",
    "INSTANCE OF": "wdt:P31",
    "HUMAN": "wd:Q5",
    "PATENT": "wd:Q253623",
    "COUNTRY": "wd:Q6256",
    "CLINICAL TRIAL": "wd:Q30612",
    "STATE": "wd:Q7275",
    "GEOGRAPHICAL FEATURE": "wd:Q618123",
    "COMICS": "wd:Q1004",
    "VOLUME": "wd:Q1238720",
    "LABEL_SERVICE_CALL": """SERVICE wikibase:label {
        bd:serviceParam wikibase:language "en"
        }
        """,
}


def __accepted_paths(accepted_properties: Iterable) -> str:
    """
    Returns the string of the conjunctive set of paths starting with an accepted property followed by any number of subclass steps.
    """
    return (
        "("
        + " | ".join(
            __s("PROPERTY") + prop + f"/{__s('SUBCLASS OF')}"
            for prop in accepted_properties
        )
        + " )"
    )


def __cat(
    c1: Union[pd.Series, list],
    c2: Union[pd.Series, list],
) -> Union[pd.Series, list]:
    return list(c1) + list(c2)


def __col(
    dataset: DatasetObject,
):
    """
    Simple agnostic column listing function.
    """
    if __pd(dataset):
        return dataset.columns
    else:
        return dataset.features


def __col_to_values_string(
    column: Union[pd.Series, list], to_uri=False, prefix=""
) -> str:
    """
    Given a column, return a properly formatted VALUES string of its content. Optionally add prefix or convert strings to URI format.
    """
    vals = __unpack_column(column)
    return f"{{ {'<'*to_uri + prefix + (stringsafe('>'*to_uri + ' ' + ' '*to_uri + prefix)).join(vals)}{'>'*to_uri} }}"


def __filter(role: str, class_: str, bidirectional=False):
    """
    Return a FILTER string against the role (e.g. subject, object) belonging to the class cls, where cls is a valid class string in __IMPORTANT_STRINGS.
    """
    return f'FILTER NOT EXISTS {{ ?{role} {__s("INSTANCE OF")}/{__s("SUBCLASS OF")} {__s(class_)} }}'


def __hf(dataset: DatasetObject) -> bool:
    """
    Convenience function for checking that a dataset is a HF dataset.
    """
    return isinstance(dataset, datasets.Dataset)


def __opp(class_: str) -> str:
    """
    Convenience function for returning the opposite role in a relation
    """
    assert class_ in ["subject", "object"]
    match class_:
        case "subject":
            return "object"
        case "object":
            return "subject"
        case _:
            raise ValueError(class_)


def __pd(dataset: DatasetObject) -> bool:
    """
    Convenience function for checking that a dataset is a DataFrame.
    """
    return isinstance(dataset, pd.DataFrame)


def __relates_to_accepted_class(
    var: str,
    accepted_properties_dict: dict[str, str],
    accepted_items: dict[str, str],
) -> str:
    """
    Returns the string corresponding to instances of the variable which relate to an accepted class by zero or more subclass relations.
    """
    return f"""?{var} {__accepted_paths(list(accepted_properties_dict.values()))} ?{var}Class .
        VALUES ?{var}Class {__to_values_string(list(accepted_items.values()), prefix=__s("ITEM"))}"""


def __acceptable_instance(var: str, accepted_items: dict[str, str]) -> str:
    """
    Returns the string corresponding to the variable being an instance of (a subclass of) an accepted item.
    """
    return f"""?{var} {__accepted_paths([__s("INSTANCE OF")[4:]])} ?{var}Class .
        VALUES ?{var}Class {__to_values_string(list(accepted_items.values()), prefix=__s("ITEM"))}"""


def __s(important_string_tag: str) -> str:
    """
    Convenience function for slightly less verbose string lookups.
    """
    return __IMPORTANT_STRINGS[important_string_tag]


def __to_values_string(itr: Iterable, prefix="") -> str:
    """
    Given a generic iterable, return a properly formatted (i.e. space-separated) VALUES string for its elements with an optional prefix.
    """
    return "{ " + " ".join((prefix + elm for elm in itr)) + " }"


def __unpack_column(column: Union[pd.Series, list]) -> list:
    """
    Extract the values from a column agnostically.
    """
    if isinstance(column, pd.Series):
        return column.values
    elif isinstance(column, list):
        return column
    else:
        raise TypeError(f"Expected pandas series or list but got {type(column)}")


def __unpack_source_id_pairs_values_string(dataset) -> str:
    """
    Given a properly formatted dataset, return a properly formatted VALUES string for pairs of source_name and source (IDs).
    """
    return (
        "{ \n"
        + "\n".join(
            (
                '( "' + _stringsafe(source) + '" ' + __s("ITEM") + sourceID + " )"
                for (source, sourceID) in zip(
                    __unpack_column(dataset["source_name"]),
                    __unpack_column(dataset["source"]),
                )
            )
        )
        + "\n}"
    )


def __unpack_website_generated_uri_pairs_value_string(dataset: DatasetObject) -> str:
    """
    Given a properly formatted dataset with source_name and website columns where multiple websites appear per source_name, unpack the values string of the pairs of these.
    """
    return f"""{{
        {'\n'.join(
            '( <' + website + '> \'' + _stringsafe(source_name) + '\' )'
            for (website, source_name) in zip(
                __unpack_column(dataset['website']),
                __unpack_column(dataset['source_name'])))}
    }}
        """


def _extract_feature_names(dataset: DatasetObject) -> list[str]:
    """
    Extract the list of feature (column) names agnostically.
    """
    if __pd(dataset):
        return dataset.columns
    elif __hf(dataset):
        return dataset.features
    else:
        raise TypeError(
            f"Expected a Pandas DataFrame or HuggingFace dataset but found {type(dataset)}"
        )


def _stringsafe(s: str) -> str:
    return stringsafe(s)


def apply_filters(
    dataset: DatasetObject,
) -> str:
    """
    Apply all filters to a SRO-dataset object and return the DataFrame of (only) ids not filtered.
    """
    validate(dataset)
    assert all(
        col in __col(dataset) for col in ["subject", "object"]
    ), f"found columns {__col(dataset)} on dataset {dataset}."
    return f"""
        SELECT ?id WHERE {{
        VALUES ?id {__col_to_values_string(__cat(dataset["subject"], dataset["object"]), prefix=__s('ITEM'))}

        SERVICE wikibase:label {{
        bd:serviceParam wikibase:language "en" .
        ?id rdfs:label ?idLabel .
        }}

        FILTER NOT EXISTS {{
            ?id wdt:P31/wdt:P279* ?filteredItem .
            VALUES ?filteredItem {{
                {__s("PATENT")}
                {__s("CLASS OF FICTIONAL ENTITIES")}
                {__s("CLINICAL TRIAL")}
                {__s("STATE")}
                {__s("GEOGRAPHICAL FEATURE")}
                {__s("COMICS")}
                {__s("VOLUME")}
            }}
        }}

        FILTER(
            !CONTAINS(LCASE(STR(?idLabel)), "wiki") &&
            !CONTAINS(LCASE(STR(?idLabel)), "wiktionary")
        )

        FILTER NOT EXISTS {{
             ?id wdt:P279 ?someclass
         }}

    }}
    """


def for_class_labels(
    dataset: DatasetObject,
    accepted_items_dict: dict[str, str],
    accepted_properties_dict: dict[str, str],
    target="source",
) -> str:
    """
    Given a properly formatted dataset, make a query for which of the accepted classes each source belongs to. Note that this behavior is inherent in many but not all of the other queries.
    """
    validate(dataset)
    assert target in _extract_feature_names(dataset)
    return f"""SELECT ?{target} ?{target}ClassLabel WHERE {{
        ?source ?wbRelation ?object .

        ?relation wikibase:directClaim ?wbRelation

        VALUES ?source {__col_to_values_string(dataset["source"], prefix=__s('ITEM'))} .

        VALUES ?wbRelation {__to_values_string(accepted_properties_dict.values(), prefix=__s("PROPERTY"))} .

        {__relates_to_accepted_class(target, accepted_properties_dict, accepted_items_dict)}

        {__s("LABEL_SERVICE_CALL")}
    }}"""


def for_neighbor_organizations(
    dataset: DatasetObject,
    accepted_items_dict: dict[str, str],
    accepted_properties_dict: dict[str, str],
    seed_role: Union[Literal["subject"], Literal["object"]] = "subject",
) -> str:
    """
    Given a properly formatted dataset, construct a query for
    relational triples with seed_role in the dataset and the provided
    accepted relations to the non-seed. The object must be an instance of (a subclass of) an accepted classes. The accepted_properties are expected to link to organizations, e.g. "has part(s)".

    Parameters -
    dataset: DatasetObject accepted_items_dict
        A dict whose values are WikiData IDs of items representing
        accepted classes of objects in the relation.
    accepted_properties_dict
        A dict whose values are WikiData IDs of properties
        representing accepted relations to the objects.
    seed_role: Union[Literal["subject"], Literal["object"]]
        Whether the item we already know ("the seed", e.g. the
        labeled node from a dataset) is placed as subject or object
        of the SRO-query.
    """
    assert seed_role in ["subject", "object"]
    validate(dataset)
    return f"""SELECT ?subjectLabel ?relationLabel ?objectLabel ?subjectDescription ?objectDescription ?{seed_role}ClassLabel ?{__opp(seed_role)}ClassLabel ?subject ?relation ?object WHERE {{
        {{ ?subject ?wbRelation ?object }}

        ?relation wikibase:directClaim ?wbRelation

        VALUES ?{seed_role} {__col_to_values_string(dataset['source'], prefix=__s('ITEM'))} .

        VALUES ?wbRelation {__to_values_string(accepted_properties_dict.values(), prefix=__s('PROPERTY'))} .

        {__acceptable_instance(__opp(seed_role), accepted_items_dict)}

        {__s('LABEL_SERVICE_CALL')}
    }}"""


def for_neighbor_humans(
    dataset: DatasetObject,
    accepted_relations_dict: dict[str, str],
    accepted_properties_dict: dict[str, str],
    accepted_items_dict: dict[str, str],
    seed_role: Union[Literal["subject"], Literal["object"]] = "subject",
) -> str:
    """
    Given a properly formatted dataset, construct a query for relational
    triples with subject in the dataset and the provided accepted
    relations to accepted classes.

    Parameters - dataset: DatasetObject accepted_items_dict
        A dict whose values are WikiData IDs of items representing
        accepted classes that the target of the relation must relate to.
    accepted_items_dict
        A dict whose values are WikiData IDs of items representing
        accepted (sub)classes, such as "knowledge worker".
    accepted_relations_dict
        A dict whose values are WikiData IDs of properties representing
        accepted relations to accepted (sub)classes (e.g. "occupation",
        which could relate to a subclass of "knowledge worker")
    accepted_object_relations_dict
        A dict whose values are WikiData IDs of properties representing
        accepted relations for the target of the query to the seed of
        the query, such as "employed by".
    seed_role: Union[Literal["subject"], Literal["object"]]
        Whether the item we already know ("the seed", e.g. the labeled
        node from a dataset) is placed as subject or object of the
        SRO-query.
    """
    assert seed_role in ["subject", "object"]
    validate(dataset)
    return f"""SELECT ?subjectLabel ?relationLabel ?objectLabel ?subjectDescription ?objectDescription ?{seed_role}ClassLabel ?{__opp(seed_role)}ClassLabel ?subject ?relation ?object WHERE {{
        {{ ?subject ?wbObjectRelation ?object }}

        ?relation wikibase:directClaim ?wbObjectRelation

        VALUES ?{seed_role} {__col_to_values_string(dataset['source'], prefix=__s('ITEM'))} .

        VALUES ?wbObjectRelation {__to_values_string(accepted_relations_dict.values(), prefix=__s('PROPERTY'))} .

        {__relates_to_accepted_class(__opp(seed_role), accepted_properties_dict, accepted_items_dict)}

        {__s('LABEL_SERVICE_CALL')}
    }}"""


def for_owners_of_official_websites(
    dataset: DatasetObject,
    uri_is_inferred=True,
) -> str:
    """
    Given a properly formatted dataset, return the query for sources which are owners of the websites given in the dataset. If uri_is_inferred then the function assumes that multiple generated URIs are being tested per source_name.
    """
    validate(dataset)
    assert {"website", "source_name"}.issubset(set(_extract_feature_names(dataset)))

    if uri_is_inferred:
        return f"""SELECT DISTINCT ?website ?source ?sourceLabel ?sourceDescription ?source_name ?sourceClassLabel WHERE {{
            ?website {__s("OFFICIAL WEBSITE OF")} ?source
            VALUES (?website ?source_name) {__unpack_website_generated_uri_pairs_value_string(dataset)}
            {__s("LABEL_SERVICE_CALL")}
        }}"""
    else:
        return f"""SELECT DISTINCT ?website ?source ?sourceLabel ?sourceDescription WHERE {{
            ?website {__s("OFFICIAL WEBSITE OF")} ?source
            VALUES ?website {__col_to_values_string(dataset["source_name"], to_uri=True)}
            {__s("LABEL_SERVICE_CALL")}
        }}"""


def for_valid_sources(
    dataset: DatasetObject,
    accepted_properties_dict: dict,
    accepted_items_dict: dict,
    birthyear_threshold=1900,
) -> str:
    """
    Construct a SPARQL query which retrieves that subset of input sources which relates to the accepted superclasses by way of an accepted relation and any number of 'subclass of' steps.

    Parameters
    -
        dataset: DatasetObject
        accepted_properties_dict:
            A dict whose values are WikiData identifiers for properties indicating accepted relation-types to accepted_items by way of an arbitrary amount of subclass-of steps.
        accepted_items_dict:
            Classes that input sources are allowed to be related to subclasses of.
        birthyear_threshold: int
            An integer designating the earliest year of birth of a returned person. Set to None to skip this filter.
    """
    validate(dataset)
    assert {"source_name", "source"}.issubset(set(_extract_feature_names(dataset)))
    return f"""SELECT ?source_name ?source ?sourceLabel ?sourceDescription ?sourceClassLabel WHERE {{
        VALUES (?source_name ?source) {__unpack_source_id_pairs_values_string(dataset)}
        {__relates_to_accepted_class("source", accepted_properties_dict, accepted_items_dict)}
        {bool(birthyear_threshold)*f'''OPTIONAL {{
            ?source {__s('BORN AT')} ?dateOfBirth . hint:Prior hint:rangeSafe true.
        }}

        {__filter("source", "CLASS OF FICTIONAL ENTITIES")}

        FILTER(!BOUND(?dateOfBirth) || ?dateOfBirth > '{str(birthyear_threshold)}-01-01'^^xsd:dateTime)'''}
        {__s("LABEL_SERVICE_CALL")}
    }}"""


def validate(dataset) -> None:
    """
    Validate that a dataset meets specifications. Currently, this is not very strict, meaning that at least one canonical feature-name and one string column is present.
    """
    feature_names = _extract_feature_names(dataset)
    if __pd(dataset):
        assert pd.core.arrays.string_.StringDtype in list(dataset.dtypes)
    elif __hf(dataset):
        assert "string" in str(dataset.features.type)
    else:
        raise TypeError(dataset)
    assert any((feature_name in __CANONICAL_FEATURES for feature_name in feature_names))
