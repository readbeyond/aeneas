#!/usr/bin/env python
# coding=utf-8

# aeneas is a Python/C library and a set of tools
# to automagically synchronize audio and text (aka forced alignment)
#
# Copyright (C) 2012-2013, Alberto Pettarin (www.albertopettarin.it)
# Copyright (C) 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
# Copyright (C) 2015-2016, Alberto Pettarin (www.albertopettarin.it)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
Run all unit tests for the aeneas package.
"""

from __future__ import absolute_import
from __future__ import print_function
import glob
import os
import sys
import unittest

__author__ = "Alberto Pettarin"
__email__ = "aeneas@readbeyond.it"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
    Copyright 2015-2016, Alberto Pettarin (www.albertopettarin.it)
"""
__license__ = "GNU AGPL 3"
__status__ = "Production"
__version__ = "1.7.1"

TEST_DIRECTORY = "aeneas/tests"
MAP = {
    "fast": ("test_*.py", "test_"),
    "bench": ("bench_test_*.py", "bench_test_"),
    "long": ("long_test_*.py", "long_test_"),
    "net": ("net_test_*.py", "net_test_"),
    "tool": ("tool_test_*.py", "tool_test_")
}


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
    if ("--help" in sys.argv) or ("-h" in sys.argv):
        print("")
        print("Usage: python %s [--bench-tests|--long-tests|--net-tests|--tool-tests] [--sort] [--verbose]" % sys.argv[0])
        print("")
        sys.exit(0)

    sort_tests = ("--sort" in sys.argv) or ("-s" in sys.argv)
    verbose = ("--verbose" in sys.argv) or ("-v" in sys.argv)

    if ("--bench-tests" in sys.argv) or ("-b" in sys.argv):
        test_type = "bench"
    elif ("--long-tests" in sys.argv) or ("-l" in sys.argv):
        test_type = "long"
    elif ("--net-tests" in sys.argv) or ("-n" in sys.argv):
        test_type = "net"
    elif ("--tool-tests" in sys.argv) or ("-t" in sys.argv):
        test_type = "tool"
    else:
        test_type = "fast"

    pattern, prefix = MAP[test_type]
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
            "tests": result.testsRun,
            "errors": len(result.errors),
            "failures": len(result.failures)
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
        print("")
        sys.exit(0)
    else:
        print("[INFO] Tests completed: errors or failures found!")
        print("")
        sys.exit(1)


if __name__ == '__main__':
    main()
