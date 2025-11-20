from __future__ import annotations
from typing import Dict, List, Optional
import json
from pathlib import Path
import networkx as nx
from networkx.readwrite import json_graph

from .schemas import Entity, Relation


class GraphStore:
    """
    Simple directed multigraph over entities and relations.
    Nodes: entity_id with attributes
    Edges: relation_id + type + properties
    """

    def __init__(self):
        self.graph = nx.MultiDiGraph()

    # Persistence

    def is_empty(self) :
        return (
            self.graph.number_of_nodes() == 0
            and self.graph.number_of_edges() == 0
        )

    def to_dict(self) :
        return json_graph.node_link_data(self.graph, edges="links")

    @classmethod
    def from_dict(cls, data: Dict) :
        inst = cls()
        inst.graph = json_graph.node_link_graph(
            data, multigraph=True, directed=True, edges="links"
        )
        return inst

    def save(self, path: Path) :
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path):
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)


    # Entity operations

    def upsert_entity(self, entity: Entity) :
        self.graph.add_node(
            entity.id,
            name=entity.name,
            type=entity.type,
            properties=entity.properties,
        )

    def find_entities_by_name(self, name: str) :
        """Return entity ids whose name contains the substring (case-insensitive)."""
        name_lower = name.lower()
        matches = []
        for node_id, data in self.graph.nodes(data=True):
            if name_lower in str(data.get("name", "")).lower():
                matches.append(node_id)
        return matches

    def get_entity(self, entity_id: str) :
        if entity_id not in self.graph:
            return None
        data = self.graph.nodes[entity_id]
        return Entity(
            id=entity_id,
            name=data.get("name", ""),
            type=data.get("type", ""),
            properties=data.get("properties", {}),
        )

    # Relation operations

    def add_relation(self, rel: Relation):
        self.graph.add_edge(
            rel.source,
            rel.target,
            key=rel.id,
            type=rel.type,
            properties=rel.properties,
        )

    def get_relations_for_entity(self, entity_id: str) :
        rels: List[Relation] = []
        for u, v, k, data in self.graph.edges(
            entity_id, keys=True, data=True
        ):
            rels.append(
                Relation(
                    id=str(k),
                    source=str(u),
                    target=str(v),
                    type=data.get("type", ""),
                    properties=data.get("properties", {}),
                )
            )
        # incoming edges as well
        for u, v, k, data in self.graph.in_edges(
            entity_id, keys=True, data=True
        ):
            if u == entity_id:
                continue
            rels.append(
                Relation(
                    id=str(k),
                    source=str(u),
                    target=str(v),
                    type=data.get("type", ""),
                    properties=data.get("properties", {}),
                )
            )
        return rels

    # Neighborhood / paths

    def get_neighborhood(
        self, entity_ids: List[str], radius: int = 1, max_nodes: int = 50
    ) :
        """
        Return subgraph neighborhood around a set of entities.
        """
        sub_nodes = set()
        for eid in entity_ids:
            if eid not in self.graph:
                continue
            nodes = nx.single_source_shortest_path_length(
                self.graph.to_undirected(), eid, cutoff=radius
            ).keys()
            sub_nodes.update(nodes)

        # Limit size
        if len(sub_nodes) > max_nodes:
            sub_nodes = set(list(sub_nodes)[:max_nodes])

        subgraph = self.graph.subgraph(sub_nodes).copy()
        # flatten into serializable dict
        nodes_out = []
        edges_out = []

        for nid, data in subgraph.nodes(data=True):
            nodes_out.append(
                {
                    "id": nid,
                    "name": data.get("name"),
                    "type": data.get("type"),
                    "properties": data.get("properties", {}),
                }
            )

        for u, v, k, data in subgraph.edges(keys=True, data=True):
            edges_out.append(
                {
                    "id": str(k),
                    "source": u,
                    "target": v,
                    "type": data.get("type"),
                    "properties": data.get("properties", {}),
                }
            )

        return {"nodes": nodes_out, "edges": edges_out}

    def get_overview_subgraph(self, max_nodes: int = 20):
        """
        Return a small, representative subgraph when no specific entities match.
        Picks top-degree nodes to keep context compact.
        """
        if self.is_empty():
            return {"nodes": [], "edges": []}

        top_nodes = sorted(
            self.graph.nodes,
            key=lambda n: self.graph.degree(n),
            reverse=True,
        )[:max_nodes]
        subgraph = self.graph.subgraph(top_nodes).copy()

        nodes_out = []
        edges_out = []

        for nid, data in subgraph.nodes(data=True):
            nodes_out.append(
                {
                    "id": nid,
                    "name": data.get("name"),
                    "type": data.get("type"),
                    "properties": data.get("properties", {}),
                }
            )

        for u, v, k, data in subgraph.edges(keys=True, data=True):
            edges_out.append(
                {
                    "id": str(k),
                    "source": u,
                    "target": v,
                    "type": data.get("type"),
                    "properties": data.get("properties", {}),
                }
            )

        return {"nodes": nodes_out, "edges": edges_out}

    def find_paths_between(
        self, source_ids: List[str], target_ids: List[str], max_paths: int = 5
    ) :
        """
        Simple shortest paths between sets of entities.
        """
        paths: List[List[str]] = []
        ug = self.graph.to_undirected()
        for s in source_ids:
            for t in target_ids:
                if s not in ug or t not in ug:
                    continue
                try:
                    path = nx.shortest_path(ug, s, t)
                    paths.append(path)
                    if len(paths) >= max_paths:
                        return paths
                except nx.NetworkXNoPath:
                    continue
        return paths
