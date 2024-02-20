cp ./scripts/train_script_qmapper.py .
mkdir -p train_out
python train_script_qmapper.py
rm -rf train_script_qmapper.py