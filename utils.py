from datetime import datetime


def generate_exp_name(args):
    exp_name = f'{args.lang}/{args.task}/{args.bert}/'
    if len(args.exp_name):
        exp_name = exp_name + f'{args.exp_name}'
    else:
        now = datetime.now().strftime("%m-%d-%H:%M")
        exp_name += now
    print(exp_name)
    return exp_name