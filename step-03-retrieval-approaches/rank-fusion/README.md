# Rank Fusion

Rank fusion for multiple runs with [Rank](https://github.com/amenra/ranx).
Supports all normalization strategies provided by ranx.
The fusion algorithms are restricted to only include those without a training or optimization step.

Normalization:
- `min-max`
- `min-max-inverted`
- `max`
- `sum`
- `zmuv`
- `rank`
- `borda`

Fusion:
- `min`
- `med`
- `anz`
- `log_isr`
- `bordafuse`
- `condorcet`
- `max`
- `sum`
- `mnz`
- `lsr`

## Development

This directory is [configured as DevContainer](https://code.visualstudio.com/docs/devcontainers/containers), i.e., you can open this directory with VS Code or some other DevContainer compatible IDE to work directly in the Docker container with all dependencies installed.

If you want to run it locally, please install the dependencies via `pip3 install -r requirements.txt`.

## Usage

```bash
python rank_fusion.py \
    <run-1-folder> \
    <run-2-folder> \
    ...
    --output <output-dir> \
```
