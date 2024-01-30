cp ./scripts/onnx_inference.py .
mkdir -p train_out
python onnx_inference.py
rm -rf onnx_inference.py