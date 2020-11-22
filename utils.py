from datetime import datetime

SEP_CODE = "\u241E"


def generate_exp_name(args):
    exp_name = f'{args.lang}/{args.model}/'
    now = datetime.now().strftime("%m-%d-%H:%M")
    exp_name += now
    if len(args.exp_name):
        exp_name = exp_name + f'_{args.exp_name}'
    return exp_name