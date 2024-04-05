import qmapper.codegen.tvm_transform as tvm_transform
import tvm

tvm_transform.operator_export('mm', (1024,1024,1024,'float32'), tvm.target.Target(target='llvm', host='llvm'))