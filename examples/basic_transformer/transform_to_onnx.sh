cp ./scripts/to_onnx.py .
mkdir -p train_out
python to_onnx.py
rm -rf to_onnx.py