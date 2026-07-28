"""``mv.struct`` — the v0.1 name for what is now :mod:`matverse.pp`.

Structure operations turned out to be the first half of preprocessing rather
than a namespace of their own: standardising a cell, screening out broken
geometries and dropping duplicate candidates are one stage of work, and the
scanpy-shaped names (``qc``, ``filter_materials``, ``filter_elements``) only
read correctly beside the structure operations they follow.

These are re-exports, not copies — same functions, same registry entries.
"""

from __future__ import annotations

from .pp import describe, standardize, supercell

__all__ = ["standardize", "supercell", "describe"]
