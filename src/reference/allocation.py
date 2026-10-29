"""Affectation par priorité et diffusion de revendications entre voisins directs.

Les positions distantes et les coûts des autres robots ne sont jamais transmis.
La résolution est garantie seulement pour le graphe statique connecté et les
identifiants uniques de ce protocole ; elle ne garantit pas des trajets sûrs.
"""
from dataclasses import dataclass
import hashlib
import math

import numpy as np


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} : entier supérieur ou égal à {minimum} requis')


@dataclass(frozen=True)
class Claim:
    owner: int
    revision: int
    slot: int

    def __post_init__(self):
        for name in ('owner', 'revision', 'slot'):
            _integer(getattr(self, name), name)


@dataclass(frozen=True)
class ClaimPacket:
    sender: int
    mission: str
    claims: tuple[Claim, ...]

    def __post_init__(self):
        _integer(self.sender, 'sender')
        claims = tuple(self.claims)
        if not isinstance(self.mission, str) or not self.mission:
            raise ValueError('Identifiant de mission requis')
        if any(not isinstance(c, Claim) for c in claims) or len({c.owner for c in claims}) != len(claims):
            raise ValueError('Revendications typées et propriétaires uniques requis')
        object.__setattr__(self, 'claims', claims)


class LocalSlotAllocator:
    """Un état par robot : consigne commune, état propre et messages reçus."""
    def __init__(self, robot_id, position, targets, *, mission_epoch=None):
        targets = np.asarray(targets, dtype=float)
        position = np.asarray(position, dtype=float)
        if targets.ndim != 2 or targets.shape[1] != 2 or len(targets) < 1 or not np.isfinite(targets).all():
            raise ValueError('Places finies (N, 2) requises')
        if len(np.unique(targets, axis=0)) != len(targets):
            raise ValueError('Places distinctes requises')
        if position.shape != (2,) or not np.isfinite(position).all():
            raise ValueError('Position propre 2D finie requise')
        _integer(robot_id, 'robot_id')
        if robot_id >= len(targets):
            raise ValueError('Identifiant robot hors effectif')
        self.robot_id = robot_id
        self.count = len(targets)
        self.mission = hashlib.sha256(np.asarray(targets, dtype='<f8').tobytes()).hexdigest()
        if mission_epoch is not None:
            if not isinstance(mission_epoch, str) or not mission_epoch:
                raise ValueError('Version de mission non vide requise')
            self.mission = hashlib.sha256(f'{self.mission}:{mission_epoch}'.encode()).hexdigest()
        # Préférences privées et figées pendant cette phase stationnaire.
        costs = np.sum((targets-position)**2, axis=1)
        self._preferences = tuple(int(i) for i in np.argsort(costs, kind='stable'))
        self._known = {robot_id: Claim(robot_id, 0, self._preferences[0])}
        self._packet_cache = None
        self._inbox_cache = None

    @property
    def slot(self):
        return self._known[self.robot_id].slot

    @property
    def knowledge_size(self):
        return len(self._known)

    @property
    def locally_consistent(self):
        """État local, sans prétendre détecter un consensus global dynamique."""
        return len(self._known) == self.count and len({c.slot for c in self._known.values()}) == self.count

    def broadcast(self):
        if self._packet_cache is None:
            self._packet_cache = ClaimPacket(self.robot_id, self.mission, tuple(self._known[k] for k in sorted(self._known)))
        return self._packet_cache

    def advance(self, packets):
        packets = tuple(packets)
        if any(not isinstance(p, ClaimPacket) for p in packets):
            raise ValueError('Paquets de revendications requis')
        if len({p.sender for p in packets}) != len(packets):
            raise ValueError('Un seul paquet par voisin et par tour requis')
        # Paquets immuables identiques : répéter la fusion est idempotent.
        # Les tours, livraisons et compteurs restent intégralement exécutés.
        if self._inbox_cache is not None and len(packets)==len(self._inbox_cache) and all(
                p is previous for p,previous in zip(packets,self._inbox_cache)):
            return self.slot
        known = dict(self._known)
        for packet in packets:
            if packet.sender >= self.count or packet.sender == self.robot_id or packet.mission != self.mission:
                raise ValueError('Émetteur ou mission incompatible')
            for claim in packet.claims:
                if claim.owner >= self.count or claim.slot >= self.count:
                    raise ValueError('Revendication hors consigne')
                current = known.get(claim.owner)
                if claim.owner == self.robot_id and claim.revision > self._known[self.robot_id].revision:
                    raise ValueError('Un relais ne peut pas réviser ma revendication')
                if current is not None and claim.revision == current.revision and claim != current:
                    raise ValueError('Deux revendications différentes pour la même révision')
                if current is None or claim.revision > current.revision:
                    known[claim.owner] = claim
        mine = known[self.robot_id]
        occupied = {c.slot for owner, c in known.items() if owner < self.robot_id}
        if mine.slot in occupied:
            slot = next(s for s in self._preferences if s not in occupied)
            known[self.robot_id] = Claim(self.robot_id, mine.revision+1, slot)
        if known!=self._known:self._packet_cache=None
        self._known = known
        self._inbox_cache = packets
        return self.slot

    def advance_current_mission(self, packets):
        """Ignorer et compter les paquets d'une ancienne version de mission."""
        packets = tuple(packets)
        if any(not isinstance(p, ClaimPacket) for p in packets):
            raise ValueError('Paquets de revendications requis')
        current = tuple(p for p in packets if p.mission == self.mission)
        slot = self.advance(current)
        return slot, len(packets)-len(current)


@dataclass(frozen=True)
class AllocationResult:
    assignments: np.ndarray
    knowledge_sizes: np.ndarray
    resolved: bool
    rounds: int
    duration_s: float
    packets_delivered: int
    claim_records_delivered: int


def run_stationary_allocation(positions, targets, adjacency, *, rounds=None, round_seconds=.1):
    """Transport synchrone stationnaire ; le graphe reste dans le simulateur.

    Un paquet ne parcourt qu'une arête par tour. Les mesures globales du résultat
    ne sont jamais renvoyées aux agents et ne déclenchent aucun arrêt anticipé.
    """
    positions, targets = np.asarray(positions, dtype=float), np.asarray(targets, dtype=float)
    graph = np.asarray(adjacency)
    if targets.ndim != 2 or targets.shape[1] != 2 or len(targets) < 1:
        raise ValueError('Places non vides (N, 2) requises')
    n = len(targets)
    if positions.shape != (n, 2) or not np.isfinite(positions).all():
        raise ValueError('Positions initiales compatibles requises')
    if graph.shape != (n, n) or graph.dtype != bool or not np.array_equal(graph, graph.T) or graph.diagonal().any():
        raise ValueError('Graphe booléen non orienté sans auto-boucle requis')
    rounds = n*n if rounds is None else rounds
    _integer(rounds, 'rounds', 1)
    if not math.isfinite(round_seconds) or round_seconds <= 0:
        raise ValueError('Durée de tour finie et positive requise')
    agents = [LocalSlotAllocator(i, positions[i], targets) for i in range(n)]
    assignments = [[a.slot for a in agents]]
    knowledge = [[a.knowledge_size for a in agents]]
    packets_delivered = records = 0
    for _ in range(rounds):
        # Capturer tous les paquets AVANT de faire avancer le premier agent.
        packets = tuple(a.broadcast() for a in agents)
        for i, agent in enumerate(agents):
            inbox = [packets[j] for j in np.flatnonzero(graph[i])]
            packets_delivered += len(inbox)
            records += sum(len(p.claims) for p in inbox)
            agent.advance(inbox)
        assignments.append([a.slot for a in agents])
        knowledge.append([a.knowledge_size for a in agents])
    resolved = (len({a.slot for a in agents}) == n and all(a.locally_consistent for a in agents)
                and len({a.broadcast().claims for a in agents}) == 1)
    return AllocationResult(np.asarray(assignments), np.asarray(knowledge), resolved, rounds,
                            rounds*round_seconds, packets_delivered, records)
