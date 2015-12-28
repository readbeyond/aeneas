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
__version__ = "1.4.0"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

TEST_DIRECTORY = "aeneas/tests"
TEST_PATTERN = "test_*.py"

class NOPStream(object):
    """ NOP stream """
    def flush(self):
        """ NOP """
        pass
    def write(self, msg):
        """ NOP """
        #print(msg)
        pass

def main():
    """ Perform tests """
    sort_tests = ("--sort" in sys.argv)
    verbose = ("--verbose" in sys.argv)

    all_files = [os.path.basename(f) for f in glob.glob(os.path.join(TEST_DIRECTORY, TEST_PATTERN))]
    cli_files = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
    selected_files = []
    for cli_file in cli_files:
        if not cli_file.startswith("test_"):
            cli_file = "test_" + cli_file
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
    nop_stream = NOPStream()
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

    if total_errors + total_failures == 0:
        print("")
        print("No errors or failures!")
        print("")



if __name__ == '__main__':
    main()



