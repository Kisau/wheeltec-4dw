"""Independent audit rejects unsafe ownership and false successful handoffs."""
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit_diverse_turn_bc import audit,Case


class EvaluationAudit(unittest.TestCase):
    def fixture(self,path):
        case=asdict(Case('RIGHT_ENTER',1,'test','nominal',0.,0.,0.,.2))
        model=path/'model.pt';model.write_bytes(b'fixture')
        e=dict(case=case,model='new',folder='episode',steps=1,bc_calls=1,success=True,reason='success',
            final_stationary=True,final_world_pose=[.9,0.,0.],disturbance_applied=False,max_retries=0,
            relocalization_frames=0,safety_intervention_frames=0,inference_p95_ms=.1)
        meta=dict(complete=True,models_unchanged=True,plan=[dict(model='new',case=case)],episodes=[e],
            models={'new':dict(path=str(model),sha256=hashlib.sha256(model.read_bytes()).hexdigest())},contract={'hashes':{}})
        (path/'evaluation.json').write_text(json.dumps(meta));folder=path/'episode';folder.mkdir()
        self.row=dict(stamp=1.,phase='TURN_LEFT',next_phase='TURN_LEFT',command=[0.,-.2],model_called=True)
        self.folder=folder;self.write_row()
        (folder/'model_calls.json').write_text(json.dumps([dict(stamp=1.,raw_angular_z=-.2,retry=0)]))
        np.savez_compressed(folder/'scans.npz',stamps=[1.],ranges=np.ones((1,10)))
        return meta
    def write_row(self):
        with gzip.open(self.folder/'decisions.jsonl.gz','wt') as f:f.write(json.dumps(self.row)+'\n')
    def test_valid_and_outside_turn_ownership(self):
        with tempfile.TemporaryDirectory() as name:
            path=Path(name);self.fixture(path)
            self.assertTrue(audit(path)['audit_passed'])
            self.row.update(phase='CENTER',next_phase='CENTER');self.write_row()
            self.assertFalse(audit(path)['audit_passed'])
    def test_false_handoff_and_wrong_runtime_action(self):
        with tempfile.TemporaryDirectory() as name:
            path=Path(name);meta=self.fixture(path)
            meta['episodes'][0]['final_world_pose']=[-.9,0.,0.]
            (path/'evaluation.json').write_text(json.dumps(meta))
            self.assertIn('0: invalid successful handoff',audit(path)['errors'])
            self.row['command'][1]=-.3;self.write_row()
            self.assertIn('0: runtime action differs from bounded policy',audit(path)['errors'])


if __name__=='__main__':unittest.main()
