"""CPU native-equivalence, geometry, ID and no-random-call replay tests."""
import copy
import json
import pickle
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch
from ultralytics.cfg import DEFAULT_CFG
from ultralytics.data.augment import Compose,Format,LetterBox,RandomFlip,RandomHSV,RandomPerspective
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import IterableSimpleNamespace
from ultralytics.utils.instance import Instances
from ultralytics.utils.ops import resample_segments,segments2boxes

from tools.exp17.engineering.augmentation import Capture,IndexIdentityResolver,ReplayPipeline,TRACE_KEY,install_native_replay,native_coordinate_points,replay_normal,support_from_trace
from tools.exp17.engineering.data import CLASSES,TrustedPairIndex,label_identity_rows


def fixture(hw=(64,64)):
    h,w=hw;image=np.arange(h*w*3,dtype=np.uint8).reshape(h,w,3)
    points=[np.array(p,np.float32) for p in (
        [[.12,.12],[.23,.12],[.23,.23],[.12,.23]],
        [[.10,.10],[.80,.10],[.80,.80],[.10,.80]],
        [[.81,.75],[.98,.75],[.98,.76],[.81,.76]])]
    labels={"img":image.copy(),"im_file":"synthetic_train_fixture.png","ori_shape":hw,
            "instances":Instances(segments2boxes(points),np.stack(resample_segments(copy.deepcopy(points),n=1000)),
                                  bbox_format="xywh",normalized=True),"cls":np.array([[0],[1],[1]],np.float32)}
    identities=[{"polygon_id":f"fixture_{i:03d}","annotation_ordinal":i,"native_label_ordinal":i,"class_id":c,
                 "normalized_native_coordinates":points[i].tolist()} for i,c in enumerate((0,1,1))]
    return image,labels,identities


def fmt():return Format(return_mask=True,mask_ratio=4,mask_overlap=True,bgr=0.)


def rng_state():
    return (random.getstate(),np.random.get_state(),torch.get_rng_state().clone(),
            [s.clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else None)


def assert_rng(test,a,b):
    test.assertEqual(a[0],b[0]);test.assertEqual(a[1][0],b[1][0]);test.assertTrue(np.array_equal(a[1][1],b[1][1]));test.assertEqual(a[1][2:],b[1][2:])
    test.assertTrue(torch.equal(a[2],b[2]));test.assertEqual(a[3] is None,b[3] is None)
    if a[3] is not None:
        test.assertEqual(len(a[3]),len(b[3]));test.assertTrue(all(torch.equal(x,y) for x,y in zip(a[3],b[3])))


def seed():random.seed(9723);np.random.seed(9724);torch.manual_seed(9725)


class FixtureResolver:
    def __init__(self,paths):self.paths={str(Path(p).resolve()):fixture()[2] for p in paths}
    def __call__(self,path):return copy.deepcopy(self.paths[str(Path(path).resolve())])


class ReplayTests(unittest.TestCase):
    def pair(self,operations,hw=(64,64),valid=None):
        image,labels,identities=fixture(hw)
        # Seed reset is only for an independent A observer equivalence test.
        # Production replay neither reads nor restores random state.
        baseline=Compose(copy.deepcopy(operations));capture=Capture(lambda _:identities)
        observed=ReplayPipeline(Compose(copy.deepcopy(operations)),capture)
        seed();a0=baseline(copy.deepcopy(labels));state0=rng_state()
        seed();a=observed(copy.deepcopy(labels));state1=rng_state();assert_rng(self,state0,state1)
        for key in ("img","masks","cls","bboxes","sem_masks","batch_idx"):
            self.assertTrue(torch.equal(a0[key],a[key]),key)
        state=rng_state()
        with patch.object(random,"random",side_effect=AssertionError("N drew Python RNG")), \
             patch.object(random,"uniform",side_effect=AssertionError("N drew Python RNG")), \
             patch.object(np.random,"uniform",side_effect=AssertionError("N drew NumPy RNG")), \
             patch.object(np.random,"random",side_effect=AssertionError("N drew NumPy RNG")), \
             patch.object(torch,"rand",side_effect=AssertionError("N drew Torch RNG")), \
             patch.object(torch,"randn",side_effect=AssertionError("N drew Torch RNG")):
            n=replay_normal(image,np.ones(hw,np.uint8) if valid is None else valid,a[TRACE_KEY])
        assert_rng(self,state,rng_state())
        for key in ("img","masks","cls","bboxes"):self.assertTrue(torch.equal(a[key],n[key]),key)
        self.assertEqual(a[TRACE_KEY].a_tensor_sha256,n["n_tensor_sha256"])
        return a,n

    def test_identity_replay(self):
        a,n=self.pair([RandomPerspective(scale=(1.,1.),translate=0.,size=(64,64)),fmt()])
        self.assertTrue(np.array_equal(a[TRACE_KEY].steps[0].params["M"],np.eye(3)))

    def test_horizontal_flip_replay(self):
        a,n=self.pair([RandomFlip(p=1.,direction="horizontal"),fmt()])
        self.assertTrue(a[TRACE_KEY].steps[0].params["flip"])

    def test_scale_replay(self):
        a,n=self.pair([RandomPerspective(scale=(1.25,1.25),translate=0.,size=(64,64)),fmt()])
        p=a[TRACE_KEY].steps[0].params;self.assertEqual(p["scale"],1.25);self.assertEqual(float(p["M"][0,0]),1.25)

    def test_translation_replay(self):
        a,n=self.pair([RandomPerspective(scale=(1.,1.),translate=.2,size=(64,64)),fmt()])
        matrix=a[TRACE_KEY].steps[0].params["M"]
        self.assertGreater(float(np.abs(matrix[:2,2]).sum()),0.)

    def test_combined_affine_replay(self):
        a,n=self.pair([RandomPerspective(scale=(.8,.8),translate=.1,degrees=12.,shear=8.,size=(64,64)),fmt()])
        matrix=a[TRACE_KEY].steps[0].params["M"]
        self.assertGreater(float(abs(matrix[0,1])+abs(matrix[1,0])),0.)

    def test_letterbox_replay(self):
        a,n=self.pair([LetterBox((64,64)),fmt()],hw=(40,80))
        p=a[TRACE_KEY].steps[0].params
        self.assertEqual(tuple(p["new_unpad"]),(64,32));self.assertEqual((p["top"],p["bottom"]),(16,16))

    def test_padding_replay(self):
        a,n=self.pair([LetterBox((64,96),scaleup=False),fmt()],hw=(32,64))
        self.assertFalse(n["valid_full"][:16].any());self.assertFalse(n["valid_full"][:,:16].any())
        self.assertTrue(n["valid_full"][16:48,16:80].all())

    def test_polygon_and_valid_mask_replay(self):
        valid=np.ones((64,64),np.uint8);valid[5:20,5:20]=0
        a,n=self.pair([RandomFlip(p=1.,direction="horizontal"),fmt()],valid=valid)
        self.assertTrue(np.array_equal(n["valid_full"],np.fliplr(valid)))
        support=support_from_trace(valid,a[TRACE_KEY])
        self.assertEqual(support["support"],n["support"])
        self.assertTrue(any(not r["support_complete"] for r in support["support"]))

    def test_image_boundary_replay_and_native_filtering(self):
        a,n=self.pair([RandomPerspective(scale=(2.,2.),translate=0.,size=(64,64)),fmt()])
        keep=a[TRACE_KEY].steps[0].keep
        self.assertTrue((~keep).any())
        self.assertTrue(all(0<=v<=64 for p in a[TRACE_KEY].initial_instances.segments for xy in p for v in xy))
        self.assertEqual(len(a[TRACE_KEY].native_target_rows),int(keep.sum()))

    def test_native_hsv_rng_and_exact_luts(self):
        a,n=self.pair([RandomPerspective(size=(64,64)),RandomHSV(.015,.7,.4),RandomFlip(.5,"horizontal"),fmt()])
        step=next(s for s in a[TRACE_KEY].steps if s.operation=="RandomHSV")
        self.assertEqual(len(step.params["luts"]),3)
        self.assertEqual(len(step.params["uniform_draw"]),3)
        compact=a[TRACE_KEY].compact_record()
        self.assertEqual(len(compact["trace_sha256"]),64)
        self.assertIn("sha256",next(s for s in compact["steps"] if s["operation"]=="RandomHSV")["params"]["luts"][0])
        json.dumps(compact,allow_nan=False)

    def test_overlap_order_keeps_original_polygon_id_not_only_class(self):
        a,n=self.pair([RandomPerspective(scale=(1.,1.),translate=0.,size=(64,64)),fmt()])
        rows=a[TRACE_KEY].native_target_rows
        self.assertEqual(rows[0]["annotation_ordinal"],1)
        self.assertEqual(rows[0]["mask_id"],1)
        self.assertEqual(rows[1]["annotation_ordinal"],0)

    def test_active_albumentations_is_a_hard_gate(self):
        from ultralytics.data.augment import Albumentations
        image,labels,identities=fixture();albu=Albumentations();albu.transform=object()
        pipeline=ReplayPipeline(Compose([albu,fmt()]),Capture(lambda _:identities))
        with self.assertRaisesRegex(ValueError,"Active Albumentations"):
            pipeline(copy.deepcopy(labels))

    def test_initial_resize_cache_and_picklable_native_collation(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);images=root/"images";labels_dir=root/"labels";images.mkdir();labels_dir.mkdir()
            image,_,identities=fixture((48,80))
            points=fixture()[2]
            for stem in ("0","1"):
                cv2.imwrite(str(images/(stem+".png")),image)
                text="".join(str(r["class_id"])+" "+" ".join(f"{v:.10f}" for p in r["normalized_native_coordinates"] for v in p)+"\n" for r in points)
                (labels_dir/(stem+".txt")).write_text(text,encoding="utf-8")
            cfg=copy.deepcopy(vars(DEFAULT_CFG));cfg.update(mosaic=0.,mixup=0.,cutmix=0.,copy_paste=0.,degrees=0.,shear=0.,perspective=0.)
            dataset=YOLODataset(img_path=str(images),imgsz=64,batch_size=2,augment=True,hyp=IterableSimpleNamespace(**cfg),
                                rect=False,cache=False,data={"names":{i:str(i) for i in range(7)},"nc":7},task="segment")
            install_native_replay(dataset,FixtureResolver(dataset.im_files))
            seed();a=dataset[0];trace=a[TRACE_KEY]
            self.assertEqual(trace.original_hw,(48,80));self.assertEqual(trace.loaded_hw,(39,64))
            self.assertEqual(len(trace.initial_resize),1)
            n=replay_normal(image,np.ones(image.shape[:2],np.uint8),trace)
            self.assertTrue(torch.equal(a["img"],n["img"]))
            cached=dataset[0];self.assertEqual(len(cached[TRACE_KEY].initial_resize),1)
            restored=pickle.loads(pickle.dumps(dataset))
            result=restored.collate_fn([restored[0],restored[1]])
            self.assertEqual(len(result[TRACE_KEY]),2)
            self.assertEqual(result["img"].shape,(2,3,64,64))

    def test_real_metadata_api_flat_coordinates_native_dataset_replay(self):
        """Exercise the production metadata API, not hand-authored Nx2 identities."""
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);images=root/"images";labels_dir=root/"labels";images.mkdir();labels_dir.mkdir()
            image,_,identities=fixture((48,80));path=(images/"synthetic.png").resolve()
            cv2.imwrite(str(path),image)
            document={"imageWidth":80,"imageHeight":48,"shapes":[
                {"shape_type":"polygon","label":CLASSES[r["class_id"]],
                 "points":[[x*80,y*48] for x,y in r["normalized_native_coordinates"]]} for r in identities]}
            native_labels=label_identity_rows("synthetic",document)
            self.assertTrue(all(np.asarray(r["normalized_native_coordinates"]).ndim==1 for r in native_labels))
            frozen_labels=copy.deepcopy(native_labels)
            (labels_dir/"synthetic.txt").write_text("\n".join(r["native_label_line"] for r in native_labels)+"\n",encoding="utf8")
            index=TrustedPairIndex([], [{"source_id":"synthetic","group_id":"synthetic_group",
                "a_path":str(path),"native_materialized_image_path":str(path),"native_labels":native_labels}])
            cfg=copy.deepcopy(vars(DEFAULT_CFG));cfg.update(mosaic=0.,mixup=0.,cutmix=0.,copy_paste=0.,degrees=0.,shear=0.,perspective=0.)
            def dataset():
                return YOLODataset(img_path=str(images),imgsz=64,batch_size=1,augment=True,hyp=IterableSimpleNamespace(**cfg),
                    rect=False,cache=False,data={"names":dict(enumerate(CLASSES)),"nc":7},task="segment")
            baseline=dataset();observed=dataset();install_native_replay(observed,IndexIdentityResolver(index))
            seed();expected=baseline[0];expected_rng=rng_state()
            seed();a=observed[0];assert_rng(self,expected_rng,rng_state())
            for key in ("img","masks","cls","bboxes","sem_masks","batch_idx"):
                self.assertTrue(torch.equal(expected[key],a[key]),key)
            state=rng_state()
            with patch.object(random,"random",side_effect=AssertionError("N drew RNG")), \
                 patch.object(np.random,"uniform",side_effect=AssertionError("N drew RNG")), \
                 patch.object(torch,"rand",side_effect=AssertionError("N drew RNG")):
                replayed=replay_normal(image,np.ones(image.shape[:2],np.uint8),a[TRACE_KEY])
            assert_rng(self,state,rng_state())
            for key in ("img","masks","cls","bboxes"):
                self.assertTrue(torch.equal(a[key],replayed[key]),key)
            self.assertEqual(native_labels,frozen_labels)

    def test_coordinate_shape_adapter_rejects_malformed_metadata(self):
        flat=np.array([.1,.2,.3,.4,.5,.6],dtype=np.float32)
        points=native_coordinate_points(flat)
        self.assertEqual(points.shape,(3,2));self.assertTrue(np.shares_memory(flat,points))
        for invalid in ([.1,.2,.3],[[.1,.2,.3],[.4,.5,.6]],[],[1,2,3,4,float("nan"),6],
                        [1,2,3,4,float("inf"),6],[True,False,True,False,True,False]):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):native_coordinate_points(invalid)


if __name__=="__main__":unittest.main()
