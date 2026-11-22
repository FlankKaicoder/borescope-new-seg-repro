"""Observe native 8.4.117 augmentation and replay it without random draws.

A executes the original native methods once. Observers retain their actual
parameters, perspective keep mask and Format's actual overlap permutation.
N uses those parameters and native deterministic apply_* methods; it never
calls stochastic get_params, samples another image, or saves/restores RNG.
Albumentations must be absent, as in the audited runtime; an installed active
Albumentations pipeline is a hard unsupported-operation gate, never skipped.
"""
import copy
import hashlib
import inspect
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics.data import augment as native
from ultralytics.data.base import BaseDataset
from ultralytics.utils.instance import Instances

TRACE_KEY = "j2_replay"


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def native_coordinate_points(value):
    """View frozen flat label coordinates as native float32 Nx2 Polygon points.

    label_identity_rows deliberately preserves the flat YOLO line ordering.
    Nx2 inputs are also supported, but malformed geometry is never repaired,
    clipped, reordered or written back to the immutable identity record.
    """
    points=np.asarray(value)
    require(points.dtype.kind in "fiu" and points.ndim in (1,2),
            "Frozen native Polygon coordinates must be numeric flat pairs or Nx2")
    if points.ndim==1:
        require(points.size>=6 and points.size%2==0,"Frozen native Polygon flat coordinate count is invalid")
        points=points.reshape(-1,2)
    require(points.ndim==2 and points.shape[1]==2 and points.shape[0]>=3,
            "Frozen native Polygon coordinates must have at least three Nx2 points")
    with np.errstate(over="ignore",invalid="ignore"):
        points=points.astype(np.float32,copy=False)
    require(np.isfinite(points).all(),"Frozen native Polygon coordinates are nonfinite")
    return points


def array_sha(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    value = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(value.dtype).encode()); h.update(str(value.shape).encode()); h.update(value.tobytes())
    return h.hexdigest()


def record(value, arrays="full"):
    if isinstance(value, np.ndarray):
        if arrays == "hash":
            return {"dtype":str(value.dtype),"shape":list(value.shape),"sha256":array_sha(value)}
        return {"dtype":str(value.dtype),"shape":list(value.shape),"values":value.tolist()}
    if isinstance(value, np.generic):return value.item()
    if isinstance(value, dict):return {str(k):record(v,arrays) for k,v in value.items()}
    if isinstance(value, (tuple,list)):return [record(v,arrays) for v in value]
    return value


def framework_identity():
    require(ultralytics.__version__ == "8.4.117", "Replay requires pinned Ultralytics 8.4.117")
    from ultralytics.data.dataset import YOLODataset
    modules = {"augment":native,"base":BaseDataset,"dataset":YOLODataset,"instances":Instances}
    return {k:hashlib.sha256(Path(inspect.getfile(v)).read_bytes()).hexdigest() for k,v in modules.items()}


@dataclass
class ReplayStep:
    operation: str
    params: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    input_hw: tuple = ()
    output_hw: tuple = ()
    keep: np.ndarray | None = None
    order: np.ndarray | None = None


@dataclass
class ReplayTrace:
    im_file: str
    original_hw: tuple
    loaded_hw: tuple
    initial_resize: list
    initial_instances: Instances
    initial_classes: np.ndarray
    initial_identities: list
    framework_sha256: dict
    steps: list = field(default_factory=list)
    native_target_rows: list = field(default_factory=list)
    native_masks: np.ndarray | None = None
    native_classes: np.ndarray | None = None
    native_boxes: np.ndarray | None = None
    a_tensor_sha256: str | None = None

    def to_record(self, arrays="full"):
        inst = self.initial_instances
        return {"schema":"exp17_native_explicit_replay_v1","im_file":self.im_file,
                "original_hw":list(self.original_hw),"loaded_hw":list(self.loaded_hw),
                "initial_resize":record(self.initial_resize,arrays),
                "initial_instances":{"bboxes":record(inst.bboxes,arrays),"segments":record(inst.segments,arrays),
                                     "bbox_format":inst._bboxes.format,"normalized":inst.normalized},
                "initial_classes":record(self.initial_classes,arrays),"initial_identities":self.initial_identities,
                "framework_sha256":self.framework_sha256,
                "steps":[{"operation":s.operation,"params":record(s.params,arrays),"config":record(s.config,arrays),
                          "input_hw":list(s.input_hw),"output_hw":list(s.output_hw),
                          "keep":record(s.keep,arrays),"order":record(s.order,arrays)} for s in self.steps],
                "native_target_rows":self.native_target_rows,"native_masks":record(self.native_masks,arrays),
                "native_classes":record(self.native_classes,arrays),"native_boxes":record(self.native_boxes,arrays),
                "a_tensor_sha256":self.a_tensor_sha256}

    @property
    def sha256(self):
        return hashlib.sha256(json.dumps(self.to_record("hash"),sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

    def compact_record(self):
        """Complete operation parameters plus array fingerprints, without 1000-point copies."""
        steps=[]
        for step in self.steps:
            params=copy.deepcopy(step.params)
            if "luts" in params:params["luts"]=[record(x,"hash") for x in params["luts"]]
            steps.append({"operation":step.operation,"params":record(params),"config":record(step.config),
                          "input_hw":list(step.input_hw),"output_hw":list(step.output_hw),
                          "keep":record(step.keep),"order":record(step.order)})
        return {"schema":"exp17_native_explicit_replay_compact_v1","trace_sha256":self.sha256,
                "im_file":self.im_file,"original_hw":list(self.original_hw),"loaded_hw":list(self.loaded_hw),
                "initial_resize":record(self.initial_resize),"framework_sha256":self.framework_sha256,
                "initial_native_segments":record(self.initial_instances.segments,"hash"),
                "initial_native_boxes":record(self.initial_instances.bboxes,"hash"),"steps":steps,
                "native_target_rows":[{k:v for k,v in r.items() if k!="normalized_native_coordinates"} for r in self.native_target_rows],
                "native_masks":record(self.native_masks,"hash"),"native_boxes":record(self.native_boxes,"hash"),
                "native_classes":record(self.native_classes,"hash"),"a_tensor_sha256":self.a_tensor_sha256}


class Capture:
    """One synchronous transform invocation per worker process."""
    def __init__(self, identity_resolver):
        self.identity_resolver=identity_resolver
        self.framework=framework_identity()
        self.trace=None;self.identities=[];self.pending_load=None;self.resize_cache={}

    def begin(self, labels):
        require(self.trace is None,"Reentrant native augmentation capture is unsupported")
        identities=copy.deepcopy(list(self.identity_resolver(labels["im_file"])))
        classes=labels["cls"].reshape(-1)
        require(len(identities)==len(classes) and all(int(r["class_id"])==int(c) for r,c in zip(identities,classes)),
                "Frozen original label ordinal/class mapping differs from native Instances")
        require(len({r["polygon_id"] for r in identities})==len(identities),"Duplicate original Polygon identity")
        for ordinal,row in enumerate(identities):
            row.setdefault("native_label_ordinal",ordinal)
            require("annotation_ordinal" in row,"Original professional JSON ordinal is required")
        if identities and all("normalized_native_coordinates" in r for r in identities):
            from ultralytics.utils.ops import resample_segments
            expected=np.stack(resample_segments([native_coordinate_points(r["normalized_native_coordinates"])
                                                 for r in identities],n=labels["instances"].segments.shape[1]))
            require(labels["instances"].normalized and np.array_equal(expected,labels["instances"].segments),
                    "Native resampled segment rows differ from frozen label-ordinal geometry")
        loaded=tuple(labels["img"].shape[:2]); original=tuple(labels.get("ori_shape",loaded))
        prefix=copy.deepcopy(self.pending_load or [])
        self.pending_load=None
        require(original==loaded or prefix,"Initial native image resize was not observed")
        self.trace=ReplayTrace(str(labels["im_file"]),original,loaded,prefix,
                               copy.deepcopy(labels["instances"]),labels["cls"].copy(),identities,self.framework)
        self.identities=identities

    def add(self, name, params, config, labels):
        require(self.trace is not None,"Native transform called outside ReplayTrace")
        s=ReplayStep(name,copy.deepcopy(params),copy.deepcopy(config),tuple(labels["img"].shape[:2]))
        self.trace.steps.append(s)
        return s

    def finish(self, labels):
        trace=self.trace
        require(trace is not None and len(self.identities)==len(labels["cls"]),"Native target identity count differs after Format")
        trace.native_target_rows=[{**r,"native_ordinal":i,"mask_id":i+1} for i,r in enumerate(self.identities)]
        trace.native_masks=labels["masks"].numpy().copy()
        trace.native_classes=labels["cls"].numpy().copy();trace.native_boxes=labels["bboxes"].numpy().copy()
        trace.a_tensor_sha256=array_sha(labels["img"])
        require(np.array_equal(trace.native_classes.reshape(-1),[r["class_id"] for r in self.identities]),
                "Native overlap mask/class/Polygon binding differs")
        labels[TRACE_KEY]=trace
        self.trace=None;self.identities=[]
        return labels


class _Perspective(native.RandomPerspective):
    def get_params(self, labels):
        params=super().get_params(labels)
        self._step=self.capture.add("RandomPerspective",params,{"degrees":self.degrees,"translate":self.translate,
            "scale":self.scale,"shear":self.shear,"perspective":self.perspective,"size":self.size},labels)
        return params

    def box_candidates(self,*args,**kwargs):
        keep=super().box_candidates(*args,**kwargs)
        self._step.keep=keep.copy()
        return keep

    def apply_instances(self,labels,params=None):
        before=self.capture.identities
        result=super().apply_instances(labels,params)
        keep=self._step.keep
        require(keep is not None and len(keep)==len(before),"Native perspective keep mask is missing")
        self.capture.identities=[r for r,k in zip(before,keep) if k]
        self._step.output_hw=tuple(labels["img"].shape[:2])
        return result


class _LetterBox(native.LetterBox):
    def get_params(self,labels):
        params=super().get_params(labels)
        self._step=self.capture.add("LetterBox",params,{"new_shape":self.new_shape,"auto":self.auto,
            "scale_fill":self.scale_fill,"scaleup":self.scaleup,"center":self.center,"stride":self.stride,
            "padding_value":self.padding_value,"interpolation":self.interpolation},labels)
        return params

    def apply_image(self,labels,params=None):
        result=super().apply_image(labels,params);self._step.output_hw=tuple(labels["img"].shape[:2]);return result


class _Flip(native.RandomFlip):
    def get_params(self,labels):
        params=super().get_params(labels)
        s=self.capture.add("RandomFlip",params,{"direction":self.direction,"p":self.p,"flip_idx":self.flip_idx},labels)
        s.output_hw=s.input_hw
        return params


class _HSV(native.RandomHSV):
    def apply_image(self,labels,params=None):
        luts=[];draws=[];original_lut=cv2.LUT;original_uniform=np.random.uniform
        def observe_lut(src,lut,*args,**kwargs):
            luts.append(lut.copy());return original_lut(src,lut,*args,**kwargs)
        def observe_uniform(*args,**kwargs):
            value=original_uniform(*args,**kwargs);draws.append(np.asarray(value).copy());return value
        s=self.capture.add("RandomHSV",{}, {"hgain":self.hgain,"sgain":self.sgain,"vgain":self.vgain},labels)
        with patch.object(cv2,"LUT",observe_lut),patch.object(np.random,"uniform",observe_uniform):
            result=super().apply_image(labels,params)
        require(len(luts) in (0,3) and len(draws)==(1 if luts else 0),"Unexpected native HSV random/LUT call sequence")
        s.params={"applied":bool(luts),"luts":luts,"uniform_draw":draws[0] if draws else None,
                  "sampled_hsv_gains":draws[0]*[self.hgain,self.sgain,self.vgain] if draws else None}
        s.output_hw=s.input_hw
        return result


class _Format(native.Format):
    def get_params(self,labels):
        params=super().get_params(labels)
        self._step=self.capture.add("Format",{}, {"bbox_format":self.bbox_format,"normalize":self.normalize,
            "return_mask":self.return_mask,"mask_ratio":self.mask_ratio,"mask_overlap":self.mask_overlap,
            "batch_idx":self.batch_idx,"bgr":self.bgr},labels)
        require(self.return_mask and self.mask_overlap and self.mask_ratio==4,"Replay requires native stride4 overlap Format")
        self._step.output_hw=self._step.input_hw
        return params

    def _format_segments(self,instances,cls,w,h):
        original=native.polygons2masks_overlap;orders=[]
        def observe(*args,**kwargs):
            result=original(*args,**kwargs);orders.append(result[1].copy());return result
        with patch.object(native,"polygons2masks_overlap",observe):
            result=super()._format_segments(instances,cls,w,h)
        require(len(orders)==1,"Unexpected native overlap raster call sequence")
        self._step.order=orders[0]
        self.capture.identities=[self.capture.identities[int(i)] for i in orders[0]]
        return result

    def _format_img(self,img):
        original=native.random.uniform;draws=[]
        def observe(*args,**kwargs):
            value=original(*args,**kwargs);draws.append(value);return value
        with patch.object(native.random,"uniform",observe):result=super()._format_img(img)
        require(len(draws)==1,"Unexpected native Format channel random draw sequence")
        self._step.params={"channel_uniform_draw":draws[0],"reverse_bgr_to_rgb":bool(draws[0]>self.bgr and img.shape[-1]==3)}
        return result


class _Albumentations(native.Albumentations):
    def __call__(self,labels):
        require(self.transform is None,"Active Albumentations is not in the audited replay runtime; do not skip it")
        s=self.capture.add("Albumentations",{"active":False,"reason":"NATIVE_OPTIONAL_DEPENDENCY_ABSENT"},{"p":self.p},labels)
        result=super().__call__(labels);s.output_hw=s.input_hw;return result


class _Disabled:
    def __init__(self,transform,capture):self.native=transform;self.capture=capture
    def __call__(self,labels):
        require(self.native.p==0,"Mosaic/MixUp/CutMix/CopyPaste must remain frozen off")
        name=type(self.native).__name__;s=self.capture.add(name,{"enabled":False},{"p":self.native.p},labels)
        image=labels["img"];classes=labels["cls"].copy();n=len(labels["instances"])
        result=self.native(labels)
        require(result["img"] is image and len(result["instances"])==n and np.array_equal(result["cls"],classes),
                "A nominally disabled mixing transform changed native input")
        s.output_hw=s.input_hw;return result


def wrap(transform,capture):
    if type(transform) is native.Compose:
        return native.Compose([wrap(t,capture) for t in transform.transforms])
    if type(transform) in (native.Mosaic,native.MixUp,native.CutMix,native.CopyPaste):return _Disabled(transform,capture)
    classes={native.RandomPerspective:_Perspective,native.LetterBox:_LetterBox,native.RandomFlip:_Flip,
             native.RandomHSV:_HSV,native.Format:_Format,native.Albumentations:_Albumentations}
    require(type(transform) in classes,"Unsupported actual native augmentation: "+type(transform).__name__)
    cls=classes[type(transform)];result=cls.__new__(cls);result.__dict__.update(vars(transform));result.capture=capture
    return result


class ReplayPipeline:
    def __init__(self,transform,capture):self.transform=wrap(transform,capture);self.capture=capture
    @property
    def transforms(self):return self.transform.transforms
    def __call__(self,labels):
        self.capture.begin(labels)
        try:return self.capture.finish(self.transform(labels))
        except Exception:
            self.capture.trace=None;self.capture.identities=[];raise


class _LoadImage:
    def __init__(self,dataset,capture):
        self.dataset=dataset;self.capture=capture;self.native=type(dataset).load_image
    def __call__(self,index,*args,**kwargs):
        observed=[];resize=cv2.resize
        def observe(img,dsize,*a,**k):
            result=resize(img,dsize,*a,**k)
            observed.append({"input_hw":tuple(img.shape[:2]),"output_hw":tuple(result.shape[:2]),
                             "dsize":tuple(dsize),"interpolation":k.get("interpolation",cv2.INTER_LINEAR)})
            return result
        with patch.object(cv2,"resize",observe):result=self.native(self.dataset,index,*args,**kwargs)
        name=str(self.dataset.im_files[index]);ori,resized=tuple(result[1]),tuple(result[2])
        if observed:self.capture.resize_cache[name]=copy.deepcopy(observed)
        elif ori!=resized:
            require(name in self.capture.resize_cache,"A pre-existing resized image cache has no captured native resize")
            observed=copy.deepcopy(self.capture.resize_cache[name])
        self.capture.pending_load=observed
        return result


def install_native_replay(dataset,identity_resolver):
    """Install after native dataset construction, before any sample is loaded.

    identity_resolver(im_file) returns full original native-label-ordinal rows,
    each including polygon_id, annotation_ordinal and class_id. Added collated
    batch['j2_replay'] is a tuple of traces; all original batch fields are kept.
    """
    require(not hasattr(dataset,"_j2_replay_capture"),"Replay capture is already installed")
    require(not any(x is not None for x in dataset.ims),"Install replay before native RAM image loading")
    capture=Capture(identity_resolver);dataset.transforms=ReplayPipeline(dataset.transforms,capture)
    dataset.load_image=_LoadImage(dataset,capture);dataset._j2_replay_capture=capture
    return capture


class IndexIdentityResolver:
    """Picklable full-668 resolver backed by the audited TrustedPairIndex."""
    def __init__(self,index):self.index=index
    def __call__(self,im_file):
        sid=self.index.source_for_image(im_file);source=self.index.train[sid]
        return [{"source_id":sid,"group_id":source["group_id"],"polygon_id":r["polygon_id"],
                 "native_label_ordinal":r["native_label_ordinal"],"annotation_ordinal":r["annotation_ordinal"],
                 "class_id":r["class_id"],"normalized_native_coordinates":r["normalized_native_coordinates"]}
                for r in source["native_labels"]]


def _native_operation(step):
    classes={"RandomPerspective":native.RandomPerspective,"LetterBox":native.LetterBox,
             "RandomFlip":native.RandomFlip,"Format":native.Format}
    return classes[step.operation](**step.config)


def _valid_operation(valid,step):
    p=step.params;name=step.operation
    if name=="LetterBox":
        if tuple(valid.shape[::-1])!=tuple(p["new_unpad"]):valid=cv2.resize(valid,tuple(p["new_unpad"]),interpolation=cv2.INTER_NEAREST)
        return cv2.copyMakeBorder(valid,p["top"],p["bottom"],p["left"],p["right"],cv2.BORDER_CONSTANT,value=0)
    if name=="RandomPerspective":
        matrix=p["M"];size=tuple(p["size"])
        if size!=tuple(valid.shape[::-1]) or (matrix!=np.eye(3)).any():
            fn=cv2.warpPerspective if step.config["perspective"] else cv2.warpAffine
            valid=fn(valid,matrix if step.config["perspective"] else matrix[:2],dsize=size,
                     flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        return valid
    if name=="RandomFlip" and p["flip"]:
        return np.flipud(valid) if p["direction"]=="vertical" else np.fliplr(valid)
    return valid


def support_from_trace(valid_u8,trace):
    """Replay only frozen Valid geometry; no N image decode or model exposure."""
    require(tuple(valid_u8.shape)==tuple(trace.original_hw) and np.isin(valid_u8,[0,1,255]).all(),"Invalid original N Valid mask")
    valid=(valid_u8>0).astype(np.uint8)
    for op in trace.initial_resize:
        require(tuple(valid.shape)==tuple(op["input_hw"]),"Initial Valid resize input differs")
        valid=cv2.resize(valid,tuple(op["dsize"]),interpolation=cv2.INTER_NEAREST)
    for step in trace.steps:valid=_valid_operation(valid,step)
    h,w=valid.shape
    require(h%4==w%4==0,"Native mask-ratio4 dimensions differ")
    valid_native=valid.reshape(h//4,4,w//4,4).all(axis=(1,3))
    masks=trace.native_masks[0]
    require(masks.shape==valid_native.shape,"Replayed Valid/native target shape differs")
    support=[{"mask_id":i+1,"native_ordinal":i,"annotation_ordinal":r["annotation_ordinal"],
              "polygon_id":r["polygon_id"],"native_mask_pixels":int((masks==i+1).sum()),
              "support_complete":bool((masks==i+1).any() and valid_native[masks==i+1].all())} for i,r in enumerate(trace.native_target_rows)]
    return {"valid_full":np.ascontiguousarray(valid),"valid_native":np.ascontiguousarray(valid_native),"support":support}


def replay_normal(normal_bgr,valid_u8,trace):
    """Deterministic CPU replay, including native Polygon filtering/raster/order.

    Input N is uint8 BGR in the original source frame. Full/stride4 Valid use
    nearest interpolation and zero padding. Stride4 validity is conservative:
    every pixel in a native 4x4 cell must be valid. Unsupported anchors are
    reported, never silently pooled over a different region.
    """
    require(normal_bgr.dtype==np.uint8 and normal_bgr.ndim==3 and normal_bgr.shape[2]==3,"Expected decoded uint8 BGR N")
    require(tuple(normal_bgr.shape[:2])==tuple(trace.original_hw),"Aligned N original frame differs from A")
    require(tuple(valid_u8.shape)==tuple(trace.original_hw) and np.isin(valid_u8,[0,1,255]).all(),"Invalid original N Valid mask")
    image=normal_bgr.copy()
    for op in trace.initial_resize:
        require(tuple(image.shape[:2])==tuple(op["input_hw"]),"Initial resize trace input differs")
        image=cv2.resize(image,tuple(op["dsize"]),interpolation=op["interpolation"])
    require(tuple(image.shape[:2])==tuple(trace.loaded_hw),"Initial resize trace output differs")
    labels={"img":image,"instances":copy.deepcopy(trace.initial_instances),"cls":trace.initial_classes.copy()}
    identities=copy.deepcopy(trace.initial_identities)
    for step in trace.steps:
        name=step.operation
        require(tuple(labels["img"].shape[:2])==tuple(step.input_hw),"Replay operation shape/order differs")
        if name in {"Mosaic","MixUp","CutMix","CopyPaste","Albumentations"}:
            require(not step.params.get("enabled",False) and not step.params.get("active",False),"An unrecorded mixed/spatial augmentation cannot be replayed")
        elif name=="RandomHSV":
            if step.params["applied"]:
                h,s,v=cv2.split(cv2.cvtColor(labels["img"],cv2.COLOR_BGR2HSV));luts=step.params["luts"]
                cv2.cvtColor(cv2.merge((cv2.LUT(h,luts[0]),cv2.LUT(s,luts[1]),cv2.LUT(v,luts[2]))),cv2.COLOR_HSV2BGR,dst=labels["img"])
        elif name=="Format":
            operation=_native_operation(step);params=operation.get_params(labels)
            image=labels["img"].transpose(2,0,1)
            if step.params["reverse_bgr_to_rgb"]:image=image[::-1]
            labels["img"]=torch.from_numpy(np.ascontiguousarray(image))
            labels=operation.apply_instances(labels,params)
            if step.order is not None:identities=[identities[int(i)] for i in step.order]
            continue
        else:
            operation=_native_operation(step)
            labels=operation.apply_image(labels,step.params)
            labels=operation.apply_instances(labels,step.params)
            if name=="RandomPerspective":
                identities=[r for r,k in zip(identities,step.keep) if k]
        require(tuple(labels["img"].shape[:2])==tuple(step.output_hw),"Replay operation output differs")
    require(isinstance(labels["img"],torch.Tensor),"Replay trace has no native Format")
    require(np.array_equal(labels["masks"].numpy(),trace.native_masks)
            and np.array_equal(labels["cls"].numpy(),trace.native_classes)
            and np.array_equal(labels["bboxes"].numpy(),trace.native_boxes),"N replay Polygon/raster/filter/order diverged from native A")
    require([r["polygon_id"] for r in identities]==[r["polygon_id"] for r in trace.native_target_rows],"N replay Polygon identity/order differs")
    support=support_from_trace(valid_u8,trace)
    return {"img":labels["img"],"masks":labels["masks"],"cls":labels["cls"],"bboxes":labels["bboxes"],
            **support,"native_target_rows":trace.native_target_rows,"trace_sha256":trace.sha256,
            "a_tensor_sha256":trace.a_tensor_sha256,"n_tensor_sha256":array_sha(labels["img"])}
