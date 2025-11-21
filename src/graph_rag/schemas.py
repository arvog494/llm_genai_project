from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass
class DocumentChunk:
    id: str
    text: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Entity:
    id: str
    name: str
    type: str
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Relation:
    id: str
    source: str      # entity id
    target: str      # entity id
    type: str
    properties: Dict[str, Any] = field(default_factory=dict)
