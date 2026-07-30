from dataclasses import dataclass
from importlib import resources
from itertools import product
from math import isfinite
import re
import shlex
from types import MappingProxyType
from typing import Iterator, Mapping, Union

import click
import yaml


CATALOG_RESOURCE = "retrieval_hyperparameters.yml"
Scalar = Union[str, int, float, bool]


@dataclass(frozen=True)
class RetrievalHyperparameterCatalog:
    schema_version: int
    approaches: Mapping[str, Mapping[str, tuple[Scalar, ...]]]


@dataclass(frozen=True)
class RetrievalConfiguration:
    identifier: str
    parameters: Mapping[str, Scalar]


def _catalog_error(message: str) -> click.UsageError:
    return click.UsageError(
        f"Invalid retrieval hyperparameter catalog {CATALOG_RESOURCE!r}: {message}"
    )


def _require_mapping(value, location: str) -> dict:
    if not isinstance(value, dict):
        raise _catalog_error(f"{location} must be a mapping.")
    return value


def _require_keys(mapping: dict, expected: set[str], location: str) -> None:
    actual = set(mapping)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected, key=str)
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected {', '.join(map(str, unexpected))}")
        raise _catalog_error(f"{location} has {', '.join(details)}.")


def _validate_name(value, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise _catalog_error(f"{location} must be a non-empty string.")
    if not value[0].isalnum() or any(
        not (character.isalnum() or character == "-") for character in value
    ):
        raise _catalog_error(
            f"{location} {value!r} may contain only letters, numbers, and hyphens."
        )
    return value


def _validate_values(values, location: str) -> tuple[Scalar, ...]:
    if not isinstance(values, list) or not values:
        raise _catalog_error(f"{location} must be a non-empty list.")

    validated = []
    seen = set()
    for index, value in enumerate(values):
        value_location = f"{location}[{index}]"
        if value is None or not isinstance(value, (str, int, float, bool)):
            raise _catalog_error(
                f"{value_location} must be a string, number, or boolean."
            )
        if isinstance(value, float) and not isfinite(value):
            raise _catalog_error(f"{value_location} must be finite.")

        identity = (type(value), value)
        if identity in seen:
            raise _catalog_error(f"{location} contains duplicate value {value!r}.")
        seen.add(identity)
        validated.append(value)

    return tuple(validated)


def parse_retrieval_hyperparameters(text: str) -> RetrievalHyperparameterCatalog:
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise _catalog_error(f"invalid YAML: {error}") from error

    root = _require_mapping(document, "root")
    _require_keys(root, {"schema_version", "approaches"}, "root")
    if type(root["schema_version"]) is not int or root["schema_version"] != 1:
        raise _catalog_error("schema_version must be 1.")

    approaches_document = _require_mapping(root["approaches"], "approaches")
    approaches = {}
    for approach_name, approach_document in approaches_document.items():
        approach_name = _validate_name(approach_name, "approach name")
        approach = _require_mapping(
            approach_document, f"approaches.{approach_name}"
        )
        _require_keys(approach, {"parameters"}, f"approaches.{approach_name}")

        parameters_document = _require_mapping(
            approach["parameters"], f"approaches.{approach_name}.parameters"
        )
        if not parameters_document:
            raise _catalog_error(
                f"approaches.{approach_name}.parameters must not be empty."
            )

        parameters = {}
        for parameter_name, parameter_document in parameters_document.items():
            parameter_name = _validate_name(
                parameter_name, f"parameter name for {approach_name}"
            )
            location = f"approaches.{approach_name}.parameters.{parameter_name}"
            parameter = _require_mapping(parameter_document, location)
            _require_keys(parameter, {"values"}, location)
            parameters[parameter_name] = _validate_values(
                parameter["values"], f"{location}.values"
            )

        approaches[approach_name] = MappingProxyType(parameters)

    return RetrievalHyperparameterCatalog(
        schema_version=1,
        approaches=MappingProxyType(approaches),
    )


def load_retrieval_hyperparameters() -> RetrievalHyperparameterCatalog:
    try:
        catalog = resources.files("lsr_benchmark").joinpath(CATALOG_RESOURCE)
        return parse_retrieval_hyperparameters(catalog.read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as error:
        raise _catalog_error(f"could not be read: {error}") from error


def _same_scalar(left: Scalar, right: Scalar) -> bool:
    return type(left) is type(right) and left == right


def _configuration_identifier(parameters: Mapping[str, Scalar]) -> str:
    if not parameters:
        return "default"

    parts = []
    for name, value in parameters.items():
        if isinstance(value, bool):
            value_text = str(value).lower()
        else:
            value_text = str(value)
        value_slug = re.sub(r"[^A-Za-z0-9]+", "-", value_text).strip("-")
        if not value_slug:
            raise _catalog_error(
                f"parameter {name!r} value {value!r} cannot form an output identifier."
            )
        parts.append(f"{name}-{value_slug}")
    return "__".join(parts)


def _configuration(parameters: dict[str, Scalar]) -> RetrievalConfiguration:
    return RetrievalConfiguration(
        identifier=_configuration_identifier(parameters),
        parameters=MappingProxyType(parameters),
    )


def _iter_configurations(
    parameters: Mapping[str, tuple[Scalar, ...]],
) -> Iterator[RetrievalConfiguration]:
    yield _configuration({})

    variation_count = max(len(values) for values in parameters.values()) - 1
    for value_index in range(1, variation_count + 1):
        for name, values in parameters.items():
            if value_index < len(values):
                yield _configuration({name: values[value_index]})

    names = tuple(parameters)
    value_axes = tuple(parameters[name] for name in names)
    for combination in product(*value_axes):
        changed = {
            name: value
            for name, value in zip(names, combination)
            if not _same_scalar(value, parameters[name][0])
        }
        if len(changed) > 1:
            yield _configuration(changed)


def retrieval_configurations(
    catalog: RetrievalHyperparameterCatalog,
    approach: str,
    grid_size: int,
) -> tuple[RetrievalConfiguration, ...]:
    if grid_size < 1:
        raise ValueError("grid_size must be at least 1")

    parameters = catalog.approaches.get(approach)
    if parameters is None:
        return (_configuration({}),)

    selected = []
    identifiers = set()
    for configuration in _iter_configurations(parameters):
        if configuration.identifier in identifiers:
            raise _catalog_error(
                f"approach {approach!r} produces duplicate configuration identifier "
                f"{configuration.identifier!r}."
            )
        identifiers.add(configuration.identifier)
        selected.append(configuration)
        if len(selected) == grid_size:
            break
    return tuple(selected)


def render_retrieval_command(
    base_command: str, configuration: RetrievalConfiguration
) -> str:
    arguments = []
    for name, value in configuration.parameters.items():
        value_text = str(value).lower() if isinstance(value, bool) else str(value)
        arguments.extend((f"--{name}", shlex.quote(value_text)))
    return " ".join((base_command, *arguments))
