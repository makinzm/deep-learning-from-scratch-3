# Add import path for the dezero directory.

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

# __init__.pyを設定した場合
from dezero import Variable

"""
# __init__.pyを設定しない場合
from dezero.core_simple import Variable
from dezero.core_simple import setup_variable
setup_variable()
"""


# __init__.py はmodule import 時に初めに実行される
# つまり__init__.pyの名前を変えるとdezeroの中にVariableはねーよとなる.
# setup_variable()を__init__の中で実行してくれてるため,とても便利


x = Variable(np.array(1.0))
y = (x + 3) ** 2
y.backward()

print(y)
print(x.grad)