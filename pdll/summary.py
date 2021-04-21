
# Params
# FLOPs

import pdll

totle_num = 0

def _conv2d_hook(layer, inputs, output):
    '''
    '''
    o = output.shape[0] * output.shape[2] * output.shape[2]   
    k = layer.weight.numel() 
    b = 1 if layer.bias is not None else 0
    
    layer.op_num += o * (k + b)


def _bn2d_hook(layer, inputs, output):
    pass


def _maxpool2d_hook(layer, inputs, output):
    pass


def _avgpool2d_hook(layer, inputs, output):
    pass


def _relu_hook(layer, inputs, output):
    '''
    '''
    layer.op_num += inputs[0].numel()


def _linear_hook(layer, inputs, output):
    '''
    '''
    in_channels = layer.weight.shape[0]    
    layer.op_num += in_channels * output.numel()


hooks = {
    pdll.nn.Conv2d : _conv2d_hook,
    pdll.nn.Linear : _linear_hook,
    pdll.nn.BatchNorm2d : _bn2d_hook,
    pdll.nn.ReLU : _relu_hook,
    
}



def register_hook(model):
    '''
    '''
    for m in model.sublayers():
        if type(m) in hooks:
            m.register_forward_post_hook( hooks[type(m)] )
            m.register_buffer('op_num', pdll.zeros(1, dtype='float32'))
        elif type(m) in (pdll.nn.Sequential, ):
            pass
        else:
            print(f'do not support {type(m)}')
            
    print('register done...')