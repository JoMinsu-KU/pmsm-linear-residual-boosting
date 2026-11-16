"""Stage integrity and source-row parsing checks independent of training."""
from pathlib import Path
import io,json,tempfile,unittest
import numpy as np
from pmsm_repro.io import stage,verify_stage,read_json,dump_json
from pmsm_repro.data import matrix
from pmsm_repro.benchmark import endpoints

class PipelineContracts(unittest.TestCase):
    def test_completed_stage_reuse_and_changed_input_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/"model"
            def build(folder):(folder/"weights.txt").write_text("original")
            stage(out,"one",build)
            stage(out,"one",lambda out:self.fail("Completed stage was rerun"))
            with self.assertRaises(ValueError):verify_stage(out,"different")
    def test_manifest_omission_and_changed_bytes_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/"model"
            stage(out,"one",lambda folder:(folder/"weights.txt").write_text("original"))
            receipt=read_json(out/"complete.json")
            dump_json(out/"complete.json",{"input_key":"one","files":[]})
            with self.assertRaises(ValueError):verify_stage(out,"one")
            dump_json(out/"complete.json",receipt)
            (out/"weights.txt").write_text("changed")
            with self.assertRaises(ValueError):verify_stage(out,"one")
    def test_failed_stage_is_preserved_and_not_promoted(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/"model"
            def fail(folder):
                (folder/"partial.txt").write_text("partial")
                raise RuntimeError("interrupted")
            with self.assertRaises(RuntimeError):stage(out,"one",fail)
            self.assertFalse(out.exists())
            attempts=list(Path(tmp).glob("model.attempt-*"))
            self.assertEqual(len(attempts),1)
            self.assertTrue((attempts[0]/"failure.json").is_file())
    def test_csv_column_order_and_binary_stream(self):
        source=b"time,i,torque,motor_speed,coolant,rotor,stator,profile_id\n0,2,3,60,10,11,999,1\n1,4,6,120,12,13,999,1\n"
        columns=["time","i","motor_speed","torque","coolant","rotor","profile_id"]
        a=matrix(source,columns);b=matrix(io.BytesIO(source),columns)
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(a[0],[0,2,60,3,10,11,1])
    def test_actual_endpoint_union(self):
        ix=endpoints(1748,300,False)
        self.assertEqual(len(ix),768)
        self.assertEqual(ix[0],300);self.assertEqual(ix[-1],1747)
        self.assertTrue(np.all(np.diff(ix)>0))
if __name__=="__main__":unittest.main()
