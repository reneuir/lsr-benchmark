import click
import pytest

from lsr_benchmark.retrieval_hyperparameters import (
    RetrievalConfiguration,
    load_retrieval_hyperparameters,
    parse_retrieval_hyperparameters,
    render_retrieval_command,
    retrieval_configurations,
)


VALID_CATALOG = """
schema_version: 1
approaches:
  example:
    parameters:
      integer-option:
        values: [10, 20]
      mixed-option:
        values: [default, 0.5, true]
"""


def test_load_packaged_retrieval_hyperparameters():
    catalog = load_retrieval_hyperparameters()

    assert catalog.schema_version == 1
    assert tuple(catalog.approaches) == ("kannolo", "seismic", "lsp")
    assert catalog.approaches["kannolo"]["ef-search"] == (200, 50, 100, 400, 800)
    assert catalog.approaches["lsp"]["compression"] == (
        "simdbp",
        "superblock",
        "raw",
    )


def test_parse_catalog_preserves_order_and_scalar_types():
    catalog = parse_retrieval_hyperparameters(VALID_CATALOG)

    assert tuple(catalog.approaches["example"]) == (
        "integer-option",
        "mixed-option",
    )
    assert catalog.approaches["example"]["mixed-option"] == (
        "default",
        0.5,
        True,
    )


def test_parsed_catalog_is_immutable():
    catalog = parse_retrieval_hyperparameters(VALID_CATALOG)

    with pytest.raises(TypeError):
        catalog.approaches["another"] = {}
    with pytest.raises(TypeError):
        catalog.approaches["example"]["integer-option"] = (30,)


def test_configurations_are_default_first_and_round_robin():
    catalog = parse_retrieval_hyperparameters(VALID_CATALOG)

    configurations = retrieval_configurations(catalog, "example", 5)

    assert [configuration.identifier for configuration in configurations] == [
        "default",
        "integer-option-20",
        "mixed-option-0-5",
        "mixed-option-true",
        "integer-option-20__mixed-option-0-5",
    ]
    assert [dict(configuration.parameters) for configuration in configurations] == [
        {},
        {"integer-option": 20},
        {"mixed-option": 0.5},
        {"mixed-option": True},
        {"integer-option": 20, "mixed-option": 0.5},
    ]


def test_configuration_count_is_limited_by_grid_size():
    catalog = parse_retrieval_hyperparameters(VALID_CATALOG)

    configurations = retrieval_configurations(catalog, "example", 2)

    assert len(configurations) == 2


def test_unknown_approach_has_only_default_configuration():
    catalog = parse_retrieval_hyperparameters(VALID_CATALOG)

    configurations = retrieval_configurations(catalog, "unknown", 10)

    assert len(configurations) == 1
    assert configurations[0].identifier == "default"
    assert dict(configurations[0].parameters) == {}


def test_generated_configuration_is_immutable():
    catalog = parse_retrieval_hyperparameters(VALID_CATALOG)
    configuration = retrieval_configurations(catalog, "example", 2)[1]

    with pytest.raises(TypeError):
        configuration.parameters["integer-option"] = 30


def test_grid_size_must_be_positive():
    catalog = parse_retrieval_hyperparameters(VALID_CATALOG)

    with pytest.raises(ValueError, match="at least 1"):
        retrieval_configurations(catalog, "example", 0)


def test_duplicate_generated_identifiers_are_rejected():
    catalog = parse_retrieval_hyperparameters(
        VALID_CATALOG.replace(
            "values: [10, 20]",
            'values: [default, "a b", "a-b"]',
        )
    )

    with pytest.raises(click.UsageError, match="duplicate configuration identifier"):
        retrieval_configurations(catalog, "example", 5)


def test_render_command_leaves_default_unchanged():
    configuration = RetrievalConfiguration("default", {})

    assert render_retrieval_command("run $input", configuration) == "run $input"


def test_render_command_quotes_values_and_formats_booleans():
    configuration = RetrievalConfiguration(
        "custom",
        {
            "path": "value with spaces",
            "enabled": False,
            "threshold": 0.5,
        },
    )

    assert render_retrieval_command("run $input", configuration) == (
        "run $input --path 'value with spaces' --enabled false --threshold 0.5"
    )


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("approaches: [", "invalid YAML"),
        ("[]", "root must be a mapping"),
        ("schema_version: 2\napproaches: {}", "schema_version must be 1"),
        ("schema_version: true\napproaches: {}", "schema_version must be 1"),
        ("schema_version: 1", "missing approaches"),
        (
            "schema_version: 1\napproaches: {}\n1: invalid",
            "unexpected 1",
        ),
        ("schema_version: 1\napproaches: []", "approaches must be a mapping"),
        (
            """
schema_version: 1
approaches:
  example:
    parameters: {}
""",
            "parameters must not be empty",
        ),
        (
            """
schema_version: 1
approaches:
  example:
    parameters:
      invalid_name:
        values: [1]
""",
            "may contain only letters, numbers, and hyphens",
        ),
        (
            """
schema_version: 1
approaches:
  example:
    parameters:
      option:
        values: []
""",
            "must be a non-empty list",
        ),
        (
            """
schema_version: 1
approaches:
  example:
    parameters:
      option:
        values: [[nested]]
""",
            "must be a string, number, or boolean",
        ),
        (
            """
schema_version: 1
approaches:
  example:
    parameters:
      option:
        values: [null]
""",
            "must be a string, number, or boolean",
        ),
        (
            """
schema_version: 1
approaches:
  example:
    parameters:
      option:
        values: [1, 1]
""",
            "contains duplicate value 1",
        ),
        (
            """
schema_version: 1
approaches:
  example:
    parameters:
      option:
        values: [.inf]
""",
            "must be finite",
        ),
    ],
)
def test_parse_rejects_invalid_catalogs(text, message):
    with pytest.raises(click.UsageError, match=message):
        parse_retrieval_hyperparameters(text)
