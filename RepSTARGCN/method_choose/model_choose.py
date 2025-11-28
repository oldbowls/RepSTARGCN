from __future__ import print_function, division
import shutil
import inspect
from model import *
from model.layers import *

def model_choose(args, block):
    m = args.model
    if m == 'repstargcn':
        model = RepSTARGCN(num_classes=args.class_num, **args.model_param)
        shutil.copy2(inspect.getfile(RepSTARGCN), args.model_saved_name)
        shutil.copy2(inspect.getfile(cnn1x1), args.model_saved_name)
    else:
        raise (RuntimeError("No modules"))

    shutil.copy2(__file__, args.model_saved_name)
    block.log('Model load finished: ' + args.model)

    return model
