import datetime
import sys
import os


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def terminal_log():
    file_path = "./log/terminal_log/terminal_log{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    if os.path.exists(file_path):
        os.remove(file_path)
    sys.getfilesystemencoding()
    sys.stdout = Logger(file_path)
