# coding=utf-8
"""Main interface for running analysis"""
import argparse

def create(args):
    # check arguments
    # check folders
        # create folders
        # or
        # exit with error
    # create files
        # analysis file
        # training file
    raise NotImplementedError

def run(args):
    raise NotImplementedError

def train(args):
    raise NotImplementedError

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers()

create_parser = subparsers.add_parser('create')
create_parser.add_argument('--overwrite', action='store_true',
                           help='If')
create_parser.add_argument('--type', help='Type of analysis', default=None)
create_parser.add_argument('name', help='Name of analysis')

run_parser = subparsers.add_parser('run')

if __name__ == '__main__':
    args = parser.parse_args()

