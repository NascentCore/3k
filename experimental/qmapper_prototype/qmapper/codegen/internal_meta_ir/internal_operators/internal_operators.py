from .add import Internal_Add
from .div import Internal_Div
from .mul import Internal_Mul
from .sub import Internal_Sub
from .reciprocal import Internal_Reciprocal
from .pow import Internal_Pow
from .sqrt import Internal_Sqrt
from .matmul import Internal_Matmul
from .transpose import Internal_Transpose
from .view import Internal_View
from .neg import Internal_Neg
from .expand import Internal_Expand
from .log_softmax import Internal_Log_Softmax, Internal_Log_Softmax_Backward
from .sum import Internal_Sum, Internal_Sum_Dim

def dispatch_internal_operator(s: str):
    if s == 'add':
        return Internal_Add
    elif s == 'div':
        return Internal_Div
    elif s == 'mul':
        return Internal_Mul
    elif s == 'sub':
        return Internal_Sub
    elif s == 'reciprocal':
        return Internal_Reciprocal
    elif s == 'pow':
        return Internal_Pow
    elif s == 'sqrt':
        return Internal_Sqrt
    elif s == 'mm':
        return Internal_Matmul
    elif s == 'transpose':
        return Internal_Transpose
    elif s == 'view':
        return Internal_View
    elif s == 'neg':
        return Internal_Neg
    elif s == 'expand':
        return Internal_Expand
    elif s == 'log_softmax':
        return Internal_Log_Softmax
    elif s == 'log_softmax_backward_data':
        return Internal_Log_Softmax_Backward
    elif s == 'sum':
        return Internal_Sum
    elif s == 'sum_dim':
        return Internal_Sum_Dim
    else:
        raise NotImplementedError