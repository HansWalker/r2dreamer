"""Run the pinned official LeWM PushT evaluator with its released checkpoint.

Example (an isolated Python environment must already contain the official deps):
  python -m scripts.run_upstream_reference --prepare-code --python /path/to/venv/bin/python \
    --checkpoint-dir /data/lewm-pusht --dataset-root /data/stable-wm \
    --output local/reports/upstream_pusht --run

No installation or checkpoint/dataset download is implicit. --prepare-code fetches
only six small official sources. The evaluator uses CUDA and the original 50-case
PushT configuration. Completion records measured results, not a paper-score claim.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.request import urlopen

from scripts.paper_faithful_reference import LEWM_COMMIT, LEWM_HASHES

HF_COMMIT = "22b330c28c27ead4bfd1888615af1340e3fe9052"
FILES = {"module.py": LEWM_HASHES["module.py"], "jepa.py": LEWM_HASHES["jepa.py"],
         "eval.py": "6584f009b40550fa0df2f8dee22b9329fbfacdfaf74464aa4fd105f464795245",
         "config/eval/pusht.yaml": "a98551ac20a7fdf336a9c2178bb96e26e769995dc5266b09cc82e2246377c7a8",
         "config/eval/launcher/local.yaml": "1dda4add0a1632ba41458377e11897cd80ba2508483a82fb6a41f0f88a06b900",
         "config/eval/solver/cem.yaml": "797629f829c0d3ad3b9c4e42d46a87f08da0b3845bc433e353fd6274cb0574df"}
CHECKPOINT_HASHES = {"config.json": "2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09",
                     "weights.pt": "48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607"}
PROBE = """
import importlib.metadata as metadata, json
names=['torch','stable-worldmodel','stable-pretraining','hydra-core','lightning','transformers','torchvision','mujoco']
versions={}
for name in names:
 try: versions[name]=metadata.version(name)
 except metadata.PackageNotFoundError: versions[name]=None
import torch
api={}
if versions['stable-worldmodel'] is not None:
 import stable_worldmodel as swm
 api={'load_pretrained':hasattr(swm.wm.utils,'load_pretrained'),'hdf5_dataset':hasattr(swm.data,'HDF5Dataset')}
print(json.dumps({'versions':versions,'cuda_available':torch.cuda.is_available(),'api':api}))
"""
CONVERT = """
import json, sys
from pathlib import Path
import torch
import stable_pretraining as spt
from jepa import JEPA
from module import ARPredictor, Embedder, MLP
source, output=map(Path,sys.argv[1:])
cfg=json.loads((source/'config.json').read_text())
encoder=spt.backbone.utils.vit_hf(cfg['encoder']['size'],patch_size=cfg['encoder']['patch_size'],image_size=cfg['encoder']['image_size'],pretrained=False,use_mask_token=False)
clean=lambda k:{name:value for name,value in cfg[k].items() if not name.startswith('_')}
projector=lambda k:MLP(input_dim=cfg[k]['input_dim'],output_dim=cfg[k]['output_dim'],hidden_dim=cfg[k]['hidden_dim'],norm_fn=torch.nn.BatchNorm1d)
model=JEPA(encoder=encoder,predictor=ARPredictor(**clean('predictor')),action_encoder=Embedder(**clean('action_encoder')),projector=projector('projector'),pred_proj=projector('pred_proj'))
model.load_state_dict(torch.load(source/'weights.pt',map_location='cpu',weights_only=True),strict=True)
output.mkdir(parents=True,exist_ok=True)
# The official eval.py uses the modern config.json + weights.pt loader. Bind
# its Hydra targets to the pinned official implementation instead of an evolving
# copy shipped inside stable-worldmodel. This changes serialization, not tensors.
cfg['_target_']='jepa.JEPA'
for key,target in [('predictor','module.ARPredictor'),('action_encoder','module.Embedder'),('projector','module.MLP'),('pred_proj','module.MLP')]:
 cfg[key]['_target_']=target
(output/'config.json').write_text(json.dumps(cfg,indent=2))
torch.save(model.state_dict(),output/'weights.pt')
import stable_worldmodel as swm
loaded=swm.wm.utils.load_pretrained(str(output))
for key,value in model.state_dict().items():torch.testing.assert_close(loaded.state_dict()[key],value,rtol=0,atol=0)
"""


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def prepare_sources(root, download):
    result, missing = {}, []
    for name, expected in FILES.items():
        path = root / name
        url = f"https://raw.githubusercontent.com/lucas-maes/le-wm/{LEWM_COMMIT}/{name}"
        if not path.exists() and download:
            with urlopen(url, timeout=30) as response:
                data = response.read(512_001)
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"Downloaded source hash mismatch: {name}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        if not path.exists():
            missing.append(str(path))
            continue
        if digest(path) != expected:
            raise ValueError(f"Cached source hash mismatch: {path}")
        result[name] = {"sha256": expected, "url": url}
    return result, missing


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable, help="Python in a separately provisioned official runtime")
    parser.add_argument("--source-root", type=Path, default=Path("local/reference_original/lewm") / LEWM_COMMIT)
    parser.add_argument("--prepare-code", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True, help="SWM cache root; contains datasets/pusht_expert_train.h5")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="store_true", help="Convert the verified release and execute original 50-case CUDA evaluation")
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error(f"Output must be a new or empty directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    args.source_root = args.source_root.resolve()
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    args.dataset_root = args.dataset_root.resolve()
    report = {"status": "NOT_RUN", "scope": "OFFICIAL_RELEASED_CHECKPOINT_ORIGINAL_PUSHT_TASK",
              "source_commit": LEWM_COMMIT, "checkpoint_repository": "quentinll/lewm-pusht", "checkpoint_revision": HF_COMMIT,
              "limitations": ["Dependency versions are recorded, not asserted to match the publication runtime.",
                              "Dataset SHA256 is recorded at execution; no original dataset digest was published in the pinned repository.",
                              "Evaluation of a released checkpoint does not reproduce its training or establish local encoder equivalence."],
              "setup": {"environment": "uv venv --python 3.10 /path/to/isolated-env; uv pip install --python /path/to/isolated-env/bin/python 'stable-worldmodel[train,env]'",
                        "checkpoint": f"hf download quentinll/lewm-pusht --revision {HF_COMMIT} --local-dir {args.checkpoint_dir}",
                        "dataset": "Obtain pusht_expert_train.h5 from the official quentinll/lewm collection; place it in --dataset-root/datasets/.",
                        "dataset_collection": "https://huggingface.co/collections/quentinll/lewm"}}
    try:
        report["sources"], missing = prepare_sources(args.source_root, args.prepare_code)
        reasons = [f"Missing pinned source: {p}" for p in missing]
        report["checkpoint_files"] = {}
        for name, expected in CHECKPOINT_HASHES.items():
            path = args.checkpoint_dir / name
            if not path.is_file():
                reasons.append(f"Missing released checkpoint file: {path}")
            elif digest(path) != expected:
                raise ValueError(f"Released checkpoint hash mismatch: {path}")
            else:
                report["checkpoint_files"][name] = {"sha256": expected, "bytes": path.stat().st_size}
        dataset = args.dataset_root / "datasets" / "pusht_expert_train.h5"
        if not dataset.is_file():
            reasons.append(f"Missing original task dataset: {dataset}")
        probe = subprocess.run([args.python, "-c", PROBE], capture_output=True, text=True, check=False)
        if probe.returncode:
            reasons.append("Official runtime probe failed: " + probe.stderr[-3000:])
        else:
            report["runtime"] = json.loads(probe.stdout.strip().splitlines()[-1])
            if not report["runtime"]["cuda_available"]:
                reasons.append("Official evaluator requires CUDA; chosen runtime reports unavailable.")
            reasons.extend(f"Missing dependency: {name}" for name, version in report["runtime"]["versions"].items() if version is None)
            reasons.extend(f"Incompatible official runtime API: {name}" for name, present in report["runtime"].get("api", {}).items() if not present)
        report["prerequisites_missing"] = reasons
        if args.run and not reasons:
            report["dataset"] = {"path": str(dataset), "sha256": digest(dataset), "bytes": dataset.stat().st_size}
            converted = args.output / "reference_checkpoint"
            env = {**os.environ, "MUJOCO_GL": "egl", "STABLEWM_HOME": str(args.dataset_root)}
            convert = subprocess.run([args.python, "-c", CONVERT, str(args.checkpoint_dir), str(converted)], cwd=args.source_root, env=env, capture_output=True, text=True)
            (args.output / "conversion.log").write_text(convert.stdout + convert.stderr)
            convert.check_returncode()
            report["checkpoint_adapter"] = "Re-export verified tensors as config.json + weights.pt, pointing Hydra targets to pinned jepa/module classes; official load_pretrained round-trip verifies identical tensors."
            command = [args.python, "eval.py", "--config-name=pusht.yaml", f"policy={converted}",
                       f"cache_dir={args.dataset_root}", "output.filename=official_pusht_results.txt", f"hydra.run.dir={args.output / 'hydra'}"]
            report["command_argv"] = command
            with (args.output / "evaluation.log").open("w") as log:
                evaluation = subprocess.run(command, cwd=args.source_root, env=env, stdout=log, stderr=subprocess.STDOUT)
            report["returncode"] = evaluation.returncode
            results_path = args.output / "official_pusht_results.txt"
            report["results_path"] = str(results_path)
            report["status"] = "COMPLETED" if evaluation.returncode == 0 and results_path.is_file() else "FAILED"
            if results_path.is_file():
                report["results_text"] = results_path.read_text()
        elif not args.run:
            report["reason"] = "Preflight only; --run was not supplied."
    except Exception as error:
        report.update(status="FAILED", error=f"{type(error).__name__}: {error}")
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "manifest": str(args.output / "manifest.json"), "prerequisites_missing": report.get("prerequisites_missing", [])}))
    return 0 if report["status"] == "COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
