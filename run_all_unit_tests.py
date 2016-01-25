#!/usr/bin/env python
# coding=utf-8

"""
Run all unit tests.
"""

from __future__ import absolute_import
from __future__ import print_function
import glob
import os
import sys
import unittest

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
    Copyright 2015-2016, Alberto Pettarin (www.albertopettarin.it)
    """
__license__ = "GNU AGPL 3"
__version__ = "1.4.1"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

TEST_DIRECTORY = "aeneas/tests"
TEST_PATTERN = "test_*.py"
TEST_PREFIX = "test_"
LONG_PATTERN = "long_test_*.py"
LONG_PREFIX = "long_test_"
NET_PATTERN = "net_test_*.py"
NET_PREFIX = "net_test_"
TOOL_PATTERN = "tool_test_*.py"
TOOL_PREFIX = "tool_test_"

class NOPStream(object):
    """ NOP stream """
    def __init__(self, verbose=False):
        self.verbose = verbose
    def flush(self):
        """ NOP """
        pass
    def write(self, msg):
        """ NOP """
        if self.verbose:
            print(msg)

def main():
    """ Perform tests """
    sort_tests = ("--sort" in sys.argv)
    verbose = ("--verbose" in sys.argv)
    long_tests = ("--long-tests" in sys.argv)
    net_tests = ("--net-tests" in sys.argv)
    tool_tests = ("--tool-tests" in sys.argv)

    if long_tests:
        pattern = LONG_PATTERN
        prefix = LONG_PREFIX
    elif net_tests:
        pattern = NET_PATTERN
        prefix = NET_PREFIX
    elif tool_tests:
        pattern = TOOL_PATTERN
        prefix = TOOL_PREFIX
    else:
        pattern = TEST_PATTERN
        prefix = TEST_PREFIX
    all_files = [os.path.basename(f) for f in glob.glob(os.path.join(TEST_DIRECTORY, pattern))]
    cli_files = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
    selected_files = []
    for cli_file in cli_files:
        if not cli_file.startswith(prefix):
            cli_file = prefix + cli_file
        if not cli_file.endswith(".py"):
            cli_file += ".py"
        if cli_file in all_files:
            selected_files.append(cli_file)
    if len(selected_files) == 0:
        selected_files = all_files

    if sort_tests:
        selected_files = sorted(selected_files)
    verbosity = 0
    if verbose:
        verbosity = 2

    results = {}
    nop_stream = NOPStream(verbose=verbose)
    for test_file in selected_files:
        print("Running", test_file, "...")
        testsuite = unittest.TestLoader().discover(start_dir=TEST_DIRECTORY, pattern=test_file)
        result = unittest.TextTestRunner(stream=nop_stream, verbosity=verbosity).run(testsuite)
        results[test_file] = {
            "tests" : result.testsRun,
            "errors" : len(result.errors),
            "failures" : len(result.failures)
        }
    total_tests = sum([results[k]["tests"] for k in results])
    total_errors = sum([results[k]["errors"] for k in results])
    total_failures = sum([results[k]["failures"] for k in results])
    print("")
    print("Tests:    ", total_tests)
    print("Errors:   ", total_errors)
    print("Failures: ", total_failures)

    if total_errors > 0:
        print("")
        print("Errors in the following tests:")
        print("\n".join([key for key in results.keys() if results[key]["errors"] > 0]))
        print("")

    if total_failures > 0:
        print("")
        print("Failures in the following tests:")
        print("\n".join([key for key in results.keys() if results[key]["failures"] > 0]))
        print("")

    print("")
    if total_errors + total_failures == 0:
        print("[INFO] Tests completed: all passed!")
    else:
        print("[INFO] Tests completed: errors or failures found!")
    print("")



if __name__ == '__main__':
    main()



