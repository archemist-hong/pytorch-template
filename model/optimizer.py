from utils.json_parser import parse_json # json package import
from torch import optim
import adamp
import model.radam as radam

# config parsing
config = parse_json("config.json")
opt_args = config.get('optimizer').get('args')

def Adam(params):
    return optim.Adam(
        params = params,
        lr = opt_args['lr'],
        betas = opt_args['betas'],
        eps = opt_args['eps'],
        weight_decay = opt_args['weight_decay'],
        amsgrad = opt_args['amsgrad'],
        foreach = opt_args['foreach'], 
        maximize = opt_args['maximize'], 
        capturable = opt_args['capturable']
    )

def AdamP(params):
    return adamp.AdamP(
        params = params,
        lr = opt_args['lr'],
        betas = opt_args['betas'],
        eps = opt_args['eps'],
        #weight_decay = opt_args['weight_decay'],
        #amsgrad = opt_args['amsgrad'],
        #foreach = opt_args['foreach'], 
        #maximize = opt_args['maximize'], 
        #capturable = opt_args['capturable']
    )

def RAdam(params):
    return radam.RAdam(
        params = params,
        lr = opt_args['lr'],
        betas = opt_args['betas'],
        eps = opt_args['eps'],
        #weight_decay = opt_args['weight_decay'],
        #amsgrad = opt_args['amsgrad'],
        #foreach = opt_args['foreach'], 
        #maximize = opt_args['maximize'], 
        #capturable = opt_args['capturable']
    )

def AdamW(params):
    return radam.AdamW(
        params = params,
        lr = opt_args['lr'],
        betas = opt_args['betas'],
        eps = opt_args['eps'],
        #weight_decay = opt_args['weight_decay'],
        #amsgrad = opt_args['amsgrad'],
        #foreach = opt_args['foreach'], 
        #maximize = opt_args['maximize'], 
        #capturable = opt_args['capturable']
    )