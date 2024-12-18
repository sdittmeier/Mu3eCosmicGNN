#!/usr/bin/env python3

"""This module is to save the full pytorch model loaded from a checkpoint
as a Torchscript model that can be used in inference.
It can be used as:
```bash
python scripts/save_full_model.py examples/Example_1/gnn_train.yaml -o saved_onnx_files --tag v1
```
As of writing, 2024/02/27, it supports the following models:
* MetricLearning
* InteractionGNN2
"""

from __future__ import annotations

from pathlib import Path
import os
import pprint
import shutil
import subprocess
import tempfile

import torch
import torch._dynamo
import torch_geometric

import numpy as np

import yaml
from pytorch_lightning import LightningModule

from acorn import stages
from acorn.core.core_utils import find_latest_checkpoint

import click

# TODO this is switched of below (by me?)... Why???
torch.use_deterministic_algorithms(True)

@click.command()
@click.option("--config", type=str, help="configuration file", default=None)
@click.option("-c", "--checkpoint", type=str, help="checkpoint path", default=None)
@click.option("-o", "--output", type=str, help="Output path", default=".")
@click.option("--tag", type=str, default=None, help="version name")
@click.option("--stage", type=str, help="configuration file", default=None)
@click.option("--model-name", type=str, help="configuration file", default=None)
@click.option("--torch-script/--no-torch-script", default=False)
@click.option("--torch-compile/--no-torch-compile", default=False)
@click.option("--onnx/--no-onnx", default=False)
@click.option("--amp/--no-amp", default=False)
@click.option("--sigmoid/--no-sigmoid", default=False)
def main(config, checkpoint, output, tag, stage, model_name, torch_script, torch_compile, onnx, amp, sigmoid):
    pprint.pprint(locals())
    if checkpoint is not None:
        checkpoint_path = Path(checkpoint)
    else:
        config_file = Path(config)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file {config_file} not found")

        with open(config_file) as f:
            config = yaml.safe_load(f)

        stage = config["stage"]
        model = config["model"]
        checkpoint_path = Path(config["stage_dir"]) / "artifacts"

    load_model_name = model_name
    if model_name == "InteractionGNN2":
        load_model_name = "RecurrentInteractionGNN2"
    if model_name == "Filter":
        load_model_name = "JitableFilter"
    if model_name == "InteractionGNN2WithPyG":
        load_model_name = "JitableInteractionGNN2WithPyG"
    if model_name == "InteractionGNN":
        load_model_name = "RecurrentInteractionGNN"

    print(f"Loading model {load_model_name} from stage {stage}")
    lightning_model = getattr(getattr(stages, stage), load_model_name)
    if not issubclass(lightning_model, LightningModule):
        raise ValueError(f"Model {model_name} is not a LightningModule")

    # find the best checkpoint in the checkpoint path
    if isinstance(checkpoint_path, str):
        checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path {checkpoint_path} not found")

    if checkpoint_path.is_dir():
        checkpoint_path = find_latest_checkpoint(
            checkpoint_path, templates=["best*.ckpt", "*.ckpt"]
        )
        if not checkpoint_path:
            raise ValueError(f"No checkpoint found in {checkpoint_path}")

    # load the checkpoint
    print(f"Loading checkpoint from {checkpoint_path} with model {lightning_model}")
    model = lightning_model.load_from_checkpoint(checkpoint_path, map_location="cpu")
    
    # set if we use amp
    model.amp = amp
    model.do_sigmoid = sigmoid

    if sigmoid:
        model_name = model_name + "_sigmoid"

    with open("hparams.yaml", "w") as f:
        yaml.dump(model.hparams, f)

    try:
        print("edge features:", model.hparams["edge_features"])
    except:
        pass
    print("node features:", model.hparams["node_features"])

    # save for use in production environment
    out_path = Path(output)
    if not out_path.exists():
        out_path.mkdir(parents=True, exist_ok=True)

    # perform some dummy inference
    num_spacepoints = 100
    num_edges = 2000
    spacepoint_features = len(model.hparams["node_features"])

    torch.manual_seed(0)
    node_features = torch.rand(num_spacepoints, spacepoint_features).to(torch.float32).cuda()
    print(node_features.shape)
    try:
        n_edge_features = len(model.hparams["edge_features"])
        edge_features = torch.rand(num_edges, n_edge_features).cuda()
    except:
        edge_features = None

    # fix seed for torch
    edge_index = torch.randint(0, 100, (2, num_edges)).to(torch.int64).cuda()
    print(edge_index)
    print(node_features)
    #return

    # creating batch that can hold a graph
    #batch = torch_geometric.data.Data()
    #batch['r']=torch.rand(num_spacepoints).to(torch.float32).cuda()
    #batch['z']=torch.rand(num_spacepoints).to(torch.float32).cuda()
    #batch['phi']=torch.rand(num_spacepoints).to(torch.float32).cuda()
    #batch.edge_index = torch.randint(0, num_spacepoints, (2, num_edges)).to(torch.int64).cuda()
    #print(batch['r'].shape)
    #print(batch['z'].shape)
    #print(batch['phi'].shape)
    #print(batch.edge_index.shape)

    if "MetricLearning" in model_name:
        input_data = (node_features,)
        input_names = ["node_features"]
        dynamic_axes = {"node_features": {0: "num_spacepoints"}}
    elif "Filter" in model_name:
        input_data = (node_features, edge_index)
        input_names = ["node_features", "edge_index"]
        dynamic_axes = {"nodes_features": {0: "num_spacepoints"}, "edge_index": {1, "num_edges"}}
    #elif "InteractionGNN" in model_name:
    #    print("Using InteractionGNN input data")
    #    input_data = (node_features, edge_index)
    #    input_names = ["batchx", "edge_index"]
    #    dynamic_axes = {"batchx": {0: "num_spacepoints"}, "edge_index": {1, "num_edges"}}
    else:
        input_data = (node_features, edge_index,)# edge_features)
        input_names = ["x", "edge_index",]# "edge_attr"]
        dynamic_axes = {
            "x": {0: "num_spacepoints"},
            "edge_index": {1: "num_edges"},
            #"edge_attr": {0: "num_edges"},
        }

    torch.use_deterministic_algorithms(True)
    model.cuda()

    print(model)

    output = model(*input_data)
    print("output shape:", output[0].shape)
    print("output:", output[0])
    print("sucessfully run model!")

    ##########################
    # Do torch script export #
    ##########################
    if torch_script:
        with torch.jit.optimized_execution(True):
            script = model.to_torchscript(example_inputs=input_data, method='trace')

        new_output = script(*input_data)
        torch.jit.freeze(script)
        if not new_output.equal(output):
            print("WARNING!!! outputs are not identical")

        # save the model
        torch_script_path = (
            out_path / f"{stage}-{model_name}-{tag}.pt"
            if tag
            else out_path / f"{stage}-{model_name}.pt"
        )

        print(f"Saving model to {torch_script_path}")
        torch.jit.save(script, torch_script_path)
        print(f"Done saving model to {torch_script_path}")


    ###########################
    # Do torch compile export #
    ###########################
    if torch_compile:
        # def try_compile(el, *args, **kwargs):
        #     try:
        #         compiled = torch.compile(el, *args, **kwargs)
        #         print(f"Compiled {el.__repr__()[:11]}")
        #         return compiled
        #     except:
        #         print(f"Couldn't compile {el.__name__}")
        #         return el
        #
        # #torch._dynamo.config.suppress_errors = True
        # print("Try to compile parts of the model!")
        # model.edge_encoder = try_compile(model.edge_encoder, dynamic=True)
        # for i in range(len(model.edge_network)):
        #     model.edge_network[i] = try_compile(model.edge_network[i], dynamic=True)
        # for i in range(len(model.node_network)):
        #     model.node_network[i] = try_compile(model.node_network[i], dynamic=True)
        # model.edge_decoder = try_compile(model.edge_decoder, dynamic=True)
        # model.edge_output_transform = try_compile(model.edge_output_transform, dynamic=True)

        dynamic_axes_torch = {}
        for ax, info  in dynamic_axes.items():
            dynamic_axes_torch[ax] = {}
            for idx, name in info.items():
                dynamic_axes_torch[ax][idx] = torch.export.Dim(name)

        print(dynamic_axes_torch)

        torch_so_path = (
            out_path / f"{stage}-{model_name}-{tag}.so"
            if tag
            else out_path / f"{stage}-{model_name}.so"
        )

        with torch.no_grad():
            tmp_so_path = torch._export.aot_compile(
                model,
                input_data,
                dynamic_shapes=dynamic_axes_torch,
                #options={"aot_inductor.output_path": str(torch_so_path)},
            )
        tmp_cpp_file = Path(tmp_so_path.replace(".so", ".cpp"))
        assert tmp_cpp_file.exists()

        tmp_files = [ f for f in os.listdir(tmp_cpp_file.parent) if not (f in tmp_so_path.replace(".so",".o")) and f[-2:] == ".o" ]
        assert len(tmp_files) == 1
        other_obj_file = tmp_cpp_file.parent / tmp_files[0]

        torch_include = "/root/software/libtorch/include"
        torch_include2 = "/root/software/libtorch/include/torch/csrc/api/include"
        torch_libdir = "/root/software/libtorch/lib"

        cuda_include = "/root/software/cuda-12.1/include"
        cuda_libdir = "/root/software/cuda-12.1/lib64"

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)

            compile_cmd = f"g++ {tmp_cpp_file} -fPIC -Wall -std=c++20 -Wno-unused-variable -Wno-unknown-pragmas -I{torch_include} -I{torch_include2} -I{cuda_include} -mavx2 -mfma -D CPU_CAPABILITY_AVX2 -D USE_CUDA -O3 -DNDEBUG -ffast-math -fno-finite-math-only -fno-unsafe-math-optimizations -ffp-contract=off -march=native -fopenmp -D TORCH_INDUCTOR_CPP_WRAPPER -D C10_USING_CUSTOM_GENERATED_MACROS -c -o {tmp_dir / 'model.o'}"
            subprocess.run(compile_cmd.split(" ")).check_returncode()

            link_cmd = f"g++ {tmp_dir / 'model.o'} {other_obj_file} -shared -fPIC -Wall -std=c++17 -Wno-unused-variable -Wno-unknown-pragmas -L{torch_libdir} -L{cuda_libdir} -ltorch -ltorch_cpu -lgomp -lc10_cuda -lcuda -ltorch_cuda -lc10 -mavx2 -mfma -D CPU_CAPABILITY_AVX2 -D USE_CUDA -O3 -DNDEBUG -ffast-math -fno-finite-math-only -fno-unsafe-math-optimizations -ffp-contract=off -march=native -fopenmp -D TORCH_INDUCTOR_CPP_WRAPPER -D C10_USING_CUSTOM_GENERATED_MACROS -o {torch_so_path}"
            print("Link command:", link_cmd)
            subprocess.run(link_cmd.split(" ")).check_returncode()

    # try to save the model to ONNX
    if onnx:
        import onnx
        print("Trying to save the model to ONNX", onnx.__version__)
        onnx_path = (
            out_path / f"{stage}-{model_name}-{tag}.onnx"
            if tag
            else out_path / f"{stage}-{model_name}.onnx"
        )

        torch.onnx.export(
            model,
            input_data,
            onnx_path,
            verbose=False,
            input_names=input_names,
            output_names=["output"],
            dynamic_axes=dynamic_axes,
        )

        input_data = (
            input_data[0].cpu().detach().numpy(),
            input_data[1].cpu().detach().numpy(),
#            input_data[2].cpu().detach().numpy()
        )

        print("Checking the model")
        print(onnx.checker.check_model(onnx.load(onnx_path)))
        print("Model is checked")

        import onnxruntime as ort
        session = ort.InferenceSession(onnx_path)
        onnx_outputs = session.run(None, 
                              {
                                  'x': input_data[0], 
                                  'edge_index': input_data[1], 
 #                                 'edge_attr': input_data[2]
                              })
        print(len(onnx_outputs[0]))
        if not np.isclose(onnx_outputs[0], output[0].cpu().detach().numpy(), rtol=0.0, atol=1.e-3).all():
            print("WARNING      ONNX inference output is not close with 1.e-3")
            mask = ~np.isclose(onnx_outputs[0], output.cpu().detach().numpy())
            print("onnx",onnx_outputs[0][mask])
            print("ref ",output[0].cpu().detach().numpy()[mask])
        else:
            print("Output of onnx runtime inference is close to reference")
            print("onnx",onnx_outputs[0][:10])
            print("ref ",output[0].cpu().detach().numpy()[:10])

        print(f"Done saving model to {onnx_path}")


if __name__ == "__main__":
    main()