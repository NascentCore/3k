## 模型简介
原始版本的Transformer，用于测试qmapper功能和性能

## 测试qmapper导出MetaIR功能

需要修改pytorch的代码：
torch/optim/adam.py下的def _single_tensor_adam函数需要替换为qmapper/api/third_party_utils.py下的 _traceable_single_tensor_adam函数


然后，在basic_transformer目录下运行test_qmapper.sh脚本，可以得到关于这个transformer模型的meta ir的计算图，在当前目录下会保存为svg和pdf两种可视化格式
