"""Affectation stationnaire par messages, sans contrôle global ni apprentissage."""
import numpy as np
import pytest

from src.reference.allocation import Claim, ClaimPacket, LocalSlotAllocator, run_stationary_allocation
from src.reference.catalog import catalog_entries
from src.reference.shapes import named_shape


def graph(n, topology):
    g = np.ones((n, n), dtype=bool) if topology == 'complete' else np.zeros((n, n), dtype=bool)
    np.fill_diagonal(g, False)
    if topology in ('chain', 'ring'):
        for i in range(n-1): g[i, i+1] = g[i+1, i] = True
    if topology == 'ring': g[0, -1] = g[-1, 0] = True
    return g


@pytest.mark.parametrize('n', [2, 4, 12])
@pytest.mark.parametrize('topology', ['chain', 'ring', 'complete'])
def test_conflicts_resolve_on_connected_static_graphs(n, topology):
    targets = np.column_stack((np.arange(n), np.zeros(n)))
    result = run_stationary_allocation(np.zeros((n, 2)), targets, graph(n, topology))
    assert result.resolved
    assert len(np.unique(result.assignments[0])) == 1
    np.testing.assert_array_equal(np.sort(result.assignments[-1]), np.arange(n))
    assert (result.knowledge_sizes[-1] == n).all()
    assert result.duration_s == pytest.approx(n*n*.1)


def test_one_hop_per_round_without_order_dependent_relay():
    n = 5
    targets = np.column_stack((np.arange(n), np.zeros(n)))
    result = run_stationary_allocation(np.zeros((n, 2)), targets, graph(n, 'chain'), rounds=2)
    assert result.knowledge_sizes[1, 0] == 2
    assert result.knowledge_sizes[2, 0] == 3
    assert not result.resolved


def test_disconnected_graph_is_not_reported_as_consensus():
    targets = np.array([[0., 0.], [1., 0.]])
    result = run_stationary_allocation(np.zeros((2, 2)), targets, np.zeros((2, 2), dtype=bool))
    assert not result.resolved
    assert result.packets_delivered == result.claim_records_delivered == 0
    assert (result.assignments[-1] == 0).all()


def test_priority_and_revision_prevent_stale_rollback():
    targets = np.column_stack((np.arange(3), np.zeros(3)))
    agent = LocalSlotAllocator(2, (0, 0), targets)
    mission = agent.mission
    agent.advance([ClaimPacket(0, mission, (Claim(0, 0, 0),))])
    assert agent.slot == 1
    agent.advance([ClaimPacket(1, mission, (Claim(1, 1, 1),))])
    assert agent.slot == 2
    before = agent.broadcast()
    agent.advance([ClaimPacket(1, mission, (Claim(1, 0, 0),))])
    assert agent.broadcast() == before


def test_neighbor_order_does_not_change_decisions_and_packets_have_no_positions():
    targets = np.column_stack((np.arange(3), np.zeros(3)))
    first, second = (LocalSlotAllocator(2, (0, 0), targets) for _ in range(2))
    packets = [ClaimPacket(i, first.mission, (Claim(i, 0, i),)) for i in range(2)]
    first.advance(packets); second.advance(reversed(packets))
    assert first.broadcast() == second.broadcast()
    assert set(first.broadcast().__dataclass_fields__) == {'sender', 'mission', 'claims'}
    assert set(first.broadcast().claims[0].__dataclass_fields__) == {'owner', 'revision', 'slot'}


@pytest.mark.parametrize('case', ['mission', 'conflict', 'self_revision', 'outside', 'duplicate_sender'])
def test_invalid_packets_rejected_atomically(case):
    targets = np.array([[0., 0.], [1., 0.]])
    agent = LocalSlotAllocator(1, (0, 0), targets)
    before = agent.broadcast()
    packet = ClaimPacket(0, agent.mission, (Claim(0, 0, 0),))
    packets = [packet]
    if case == 'mission': packets = [ClaimPacket(0, 'other', packet.claims)]
    elif case == 'conflict': packets = [ClaimPacket(0, agent.mission, (Claim(1, 0, 1),))]
    elif case == 'self_revision': packets = [ClaimPacket(0, agent.mission, (Claim(1, 1, 1),))]
    elif case == 'outside': packets = [ClaimPacket(0, agent.mission, (Claim(0, 0, 2),))]
    else: packets = [packet, packet]
    with pytest.raises(ValueError): agent.advance(packets)
    assert agent.broadcast() == before


@pytest.mark.parametrize('identifier', [e['identifier'] for e in catalog_entries()])
def test_all_catalog_shapes_resolve_without_duplicate_places(identifier):
    targets = named_shape(identifier, 12).reference_points()
    positions = np.tile(targets[0], (12, 1))
    result = run_stationary_allocation(positions, targets, graph(12, 'chain'))
    assert result.resolved
    assert len(np.unique(result.assignments[-1])) == 12


@pytest.mark.parametrize('case', ['negative_id', 'unknown_id', 'duplicate_targets', 'nan_position', 'bad_graph', 'zero_rounds'])
def test_invalid_inputs(case):
    positions = np.zeros((2, 2)); targets = np.array([[0., 0.], [1., 0.]])
    g = graph(2, 'complete')
    with pytest.raises(ValueError):
        if case == 'negative_id': LocalSlotAllocator(-1, (0, 0), targets)
        elif case == 'unknown_id': LocalSlotAllocator(2, (0, 0), targets)
        elif case == 'duplicate_targets': LocalSlotAllocator(0, (0, 0), np.zeros((2, 2)))
        elif case == 'nan_position': LocalSlotAllocator(0, (float('nan'), 0), targets)
        elif case == 'bad_graph': run_stationary_allocation(positions, targets, g.astype(int))
        else: run_stationary_allocation(positions, targets, g, rounds=0)
