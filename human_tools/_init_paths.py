import os
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


curr_dir = os.path.dirname(os.path.abspath(__file__))
unseen_root = os.path.abspath(os.path.join(curr_dir, ".."))
lib_path = os.path.abspath(os.path.join(curr_dir, "../lib"))

# add lib to PYTHONPATH
add_path(lib_path)

# print(unseen_root)
# add_path(unseen_root)
