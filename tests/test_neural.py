"""Neural arithmetic and explicit-device tests; small CPU fits only."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from pmsm_repro import neural

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def fixture():
    rng=np.random.Generator(np.random.PCG64(42))
    x=rng.normal(size=(45,33)).astype(np.float64)
    y=20.+2*x[:,0]-x[:,1]
    p=np.r_[np.full(12,7),np.full(33,9)].astype(np.int64)
    return x,y,p


def fake_artifact():
    rng=np.random.Generator(np.random.PCG64(13))
    w={k:(rng.normal(size=s)*.02).astype(np.float32) for k,s in neural.WEIGHT_SHAPES.items()}
    return dict(weights=w,transform=dict(input_mean=[0.]*33,input_std=[1.]*33,y_mid=25.,y_scale=7.),
                metadata=dict(seed=neural.SEEDS[0],final_array_sha256=neural.array_digest(w)),trace=[])


class NeuralArithmetic(unittest.TestCase):
    def test_unweighted_statistics_balanced_loss(self):
        x,y,p=fixture()
        tr,a=neural.training_transform(x,y,p,dataset="ipmsm")
        np.testing.assert_array_equal(tr["input_mean"],x.mean(0))
        np.testing.assert_array_equal(tr["input_std"],x.std(0))
        self.assertEqual(tr["y_mid"],float(np.median(y)))
        for pid in [7,9]:self.assertAlmostEqual(float(a["w"][p==pid].sum()),len(y)/2,places=5)
        self.assertEqual(a["x"].dtype,np.float32)
        self.assertEqual(a["w"].dtype,np.float32)

    def test_floors(self):
        tr,a=neural.training_transform(np.ones((6,33)),np.ones(6),np.zeros(6,dtype=np.int64),dataset="rwth")
        self.assertEqual(tr["input_std"],[1e-8]*33)
        self.assertEqual(tr["y_scale"],1.)
        np.testing.assert_array_equal(a["x"],0.)

    def test_cpu_full_fit_refused_before_training(self):
        x,y,p=fixture()
        with self.assertRaisesRegex(ValueError,"CUDA"):
            neural.fit_neural(x,y,p,seed=neural.SEEDS[0],dataset="ipmsm",device="cpu")
        with self.assertRaises(ValueError):
            neural.fit_neural(x,y,p,seed=2026111501,dataset="ipmsm",device="cpu",epochs=1,smoke=True)

    def test_dataset_gradient_contract(self):
        self.assertIsNone(neural.settings("rwth")["gradient_norm_cap"])
        self.assertEqual(neural.settings("ipmsm")["gradient_norm_cap"],1.)
        self.assertEqual(neural.settings("paderborn")["gradient_norm_cap"],1.)

    def test_scalar_forward_and_batch_agree(self):
        artifact=fake_artifact();x,y,p=fixture()
        # Scalar is the published weight@vector path; batch is allowed FP32 GEMM rounding.
        scalar=neural.predict_numpy(artifact,x,scalar=True)
        batch=neural.predict_numpy(artifact,x)
        np.testing.assert_allclose(batch,scalar,atol=2e-6,rtol=1e-7)
        self.assertEqual(batch.dtype,np.float64)

    def test_save_load_and_weight_corruption(self):
        artifact=fake_artifact();x,_,_=fixture()
        with tempfile.TemporaryDirectory() as td:
            neural.save_neural(artifact,td)
            restored=neural.load_neural(td)
            np.testing.assert_array_equal(neural.predict_numpy(restored,x),neural.predict_numpy(artifact,x))
            with self.assertRaises(FileExistsError):neural.save_neural(artifact,td)
            altered={k:v.copy() for k,v in artifact["weights"].items()}
            altered["4.bias"][0]+=1
            np.savez_compressed(Path(td)/"weights_final.npz",**altered)
            with self.assertRaisesRegex(ValueError,"digest"):neural.load_neural(td)

    def test_input_validation(self):
        x,y,p=fixture()
        with self.assertRaises(ValueError):neural.training_transform(x,y,p,profile_order=[7])
        with self.assertRaises(ValueError):neural.training_transform(x,np.full(len(x),np.nan),p)
        with self.assertRaises(ValueError):neural.predict_neural(fake_artifact(),x,device="cuda",backend="numpy")


@unittest.skipUnless(TORCH_AVAILABLE,"PyTorch not installed; NumPy tests still run")
class NeuralCPUSmoke(unittest.TestCase):
    def test_two_epoch_cpu_smoke_and_forward_parity(self):
        x,y,p=fixture();calls=[]
        artifact=neural.fit_neural(x,y,p,seed=neural.SEEDS[0],dataset="ipmsm",device="cpu",epochs=2,
                                   smoke=True,check=lambda:calls.append(1))
        self.assertTrue(artifact["metadata"]["smoke_only"])
        self.assertFalse(artifact["metadata"]["completed_paper_fit"])
        self.assertEqual(len(artifact["trace"]),2)
        self.assertGreaterEqual(len(calls),6)
        self.assertNotEqual(artifact["metadata"]["initial_array_sha256"],artifact["metadata"]["final_array_sha256"])
        order=np.random.Generator(np.random.PCG64(neural.SEEDS[0])).permutation(len(x)).astype(np.int64)
        self.assertEqual(artifact["trace"][0]["permutation_sha256"],neural.array_digest(dict(order=order)))
        np.testing.assert_allclose(neural.predict_numpy(artifact,x),neural.predict_neural(artifact,x,backend="torch"),atol=1e-5,rtol=1e-6)

    def test_rwth_no_clip_and_paired_initialization(self):
        x,y,p=fixture()
        a=neural.fit_neural(x,y,p,seed=neural.SEEDS[1],dataset="rwth",device="cpu",epochs=1,smoke=True)
        b=neural.fit_neural(x*2,y,p,seed=neural.SEEDS[1],dataset="paderborn",device="cpu",epochs=1,smoke=True)
        self.assertIsNone(a["metadata"]["settings"]["gradient_norm_cap"])
        self.assertEqual(a["metadata"]["initial_array_sha256"],b["metadata"]["initial_array_sha256"])


if __name__=="__main__":unittest.main()
