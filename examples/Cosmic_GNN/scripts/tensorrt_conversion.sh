#!/bin/bash

MIN_NODES=1
MIN_EDGES=1

MAX_NODES=1000
MAX_EDGES=100000

NODES=80
EDGES=3000

# random input as float32
python3 -c "import numpy as np; np.random.uniform(0,1,($NODES,3)).astype(np.float32).tofile('nodes.npz')"
python3 -c "import numpy as np; np.random.randint(0,$NODES,(2,$EDGES)).astype(np.int64).tofile('edge_index.npz')"

trtexec \
    --directIO \
    --onnx=$1.onnx \
    --saveEngine=$1.engine \
    --minShapes=x:${MIN_NODES}x3,edge_index:2x${MIN_EDGES} \
    --maxShapes=x:${MAX_NODES}x3,edge_index:2x${MAX_EDGES} \
    --optShapes=x:${NODES}x3,edge_index:2x${EDGES} \
    --shapes=x:${NODES}x3,edge_index:2x${EDGES} \
    --loadInputs='x':'nodes.npz','edge_index':'edge_index.npz' \
    --verbose \
    #--inputIOFormats=fp32:chw,int64:chw

rm -f nodes.npz edge_index.npz