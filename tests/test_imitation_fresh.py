"""Campagne fraîche : protocole figé, appariement et échecs conservés."""
import hashlib
import json

import numpy as np
import pytest

from src.imitation.checkpoint import save_model
from src.imitation.evaluate_fresh import evaluate_fresh
from src.imitation.model import AccelerationGAT, GATConfig
from test_imitation_integration import source, reviewed


@pytest.fixture
def run(reviewed, tmp_path):
    model = AccelerationGAT(GATConfig.from_yaml('config/imitation-model-relative.yaml'), 2.)
    folder = tmp_path/'run'; folder.mkdir(); save_model(folder/'best.pt', model)
    (folder/'run.json').write_text(json.dumps(dict(status='trained', checkpoint='best.pt', model=model.metadata(),
        dataset_manifest_sha256=hashlib.sha256((reviewed/'manifest.json').read_bytes()).hexdigest())))
    return folder


@pytest.mark.parametrize('seeds', [[], [400, 400], [True], [-1], [1], [2], [3], [4]])
def test_invalid_or_used_seeds_rejected_before_outputs(run, reviewed, tmp_path, seeds):
    output = tmp_path/'fresh'
    with pytest.raises(ValueError):
        evaluate_fresh(run, reviewed, output, seeds)
    assert not output.exists()


def test_fresh_campaign_registers_before_execution_and_keeps_all_failures(run, reviewed, tmp_path, monkeypatch):
    import src.imitation.evaluate_fresh as fresh
    output = tmp_path/'fresh'
    original = fresh.AccelerationEpisode.execute
    calls = []
    before = (run/'best.pt').read_bytes()
    def execute(self, policy, n, seed, initial):
        protocol = json.loads((output/'protocol.json').read_text())
        assert protocol['seeds'] == [400, 401]
        assert not (output/'evaluation.json').exists()
        calls.append((seed, initial.copy()))
        return original(self, policy, n, seed, initial)
    monkeypatch.setattr(fresh.AccelerationEpisode, 'execute', execute)
    report = evaluate_fresh(run, reviewed, output, [400, 401])
    assert [r['seed'] for r in report['reports']] == [400, 401]
    assert report['status'] == 'complete'
    for i in (0, 2):
        np.testing.assert_array_equal(calls[i][1], calls[i+1][1])
    assert report['summary']['gat']['episodes'] == 2
    assert (run/'best.pt').read_bytes() == before
    with pytest.raises(FileExistsError):
        evaluate_fresh(run, reviewed, output, [400, 401])
