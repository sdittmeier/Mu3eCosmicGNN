export CUBLAS_WORKSPACE_CONFIG=:4096:8
python save_full_model.py -c test.ckpt -o output --model-name InteractionGNN --stage edge_classifier --onnx