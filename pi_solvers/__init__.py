import sys

import pi_solvers.torch_utils as torch_utils
import pi_solvers.dnnlib as dnnlib

sys.modules['torch_utils'] = torch_utils
sys.modules['dnnlib'] = dnnlib

