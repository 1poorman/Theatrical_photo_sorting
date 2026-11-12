# SwinFace inference source

Source: https://github.com/lxq1000/SwinFace at
`ca36196814b54964e663e0991efd7d163c0069d3` (2026-11-11 retrieval).

The three Python modules are copied from the upstream inference implementation.
Only CRLF normalization and the import migration from `timm.models.layers` to
`timm.layers` have been applied. The MIT license is preserved here; source paths
and original/vendored SHA256 values are recorded in `provenance.json`.

Training code, data loaders, distributed initialization and weights are not
vendored. The application wrapper loads all four official model state dicts
strictly, and evaluates only the requested identity and/or age branches.
