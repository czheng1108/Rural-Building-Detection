import os
import utils.log_utils.terminal_txt as termial

def check_dir(args):
    # check whether the save weight folder exists, create it if it does not exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists('./log'):
        os.makedirs('./log')
    if not os.path.exists('./log/terminal_log'):
        os.makedirs('./log/terminal_log')
    if not os.path.exists('./log/results_file'):
        os.makedirs('./log/results_file')
    if not os.path.exists('./log/loss_and_lr'):
        os.makedirs('./log/loss_and_lr')
    if not os.path.exists('./log/mAP'):
        os.makedirs('./log/mAP')

    termial.terminal_log()