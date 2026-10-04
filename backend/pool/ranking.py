"""Spec-based GPU allocation estimates; browser probes only fine-tune them.

Browsers usually hide GPU compute units and exact VRAM. Device class supplies
the main prior; exposed Apple model tiers and host cores/RAM add bounded hints.
These are allocation heuristics, never claimed hardware throughput measurements.
"""
import re

DEVICE_PRIORS = {'phone': 1.0, 'tablet': 2.0, 'browser': 2.0, 'laptop': 4.0, 'desktop': 5.0}


def gpu_spec_score(caps):
    if not caps.webgpu:
        return 0.0
    model = ' '.join((caps.adapter.vendor, caps.adapter.description, caps.adapter.device)).lower()
    tier = 1.0
    # Only use a tier when the browser actually supplies a recognizable model.
    match = re.search(r'\bm\d+\s*(pro|max|ultra)\b', model)
    if match:
        tier = {'pro': 1.5, 'max': 1.75, 'ultra': 2.0}[match.group(1)]
    cores = min(caps.hardware.logical_cores or 4, 32)
    memory = min(caps.hardware.memory_gib or 4, 64)
    resources = 1 + .2 * cores / 32 + .2 * memory / 64
    fallback = .1 if caps.adapter.is_fallback else 1.0
    return DEVICE_PRIORS[caps.device_type] * tier * resources * fallback
