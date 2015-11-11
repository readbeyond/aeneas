#!/usr/bin/env python
# coding=utf-8

"""
Execute a job, passed as a container or
as a container + configuration string (wizard case).
"""

import sys

from aeneas.executejob import ExecuteJob
from aeneas.logger import Logger
from aeneas.validator import Validator
import aeneas.globalfunctions as gf

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
    Copyright 2015,      Alberto Pettarin (www.albertopettarin.it)
    """
__license__ = "GNU AGPL 3"
__version__ = "1.3.2"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

NAME = "aeneas.tools.execute_job"

INPUT_FILE = gf.get_rel_path("res/job.zip")
OUTPUT_DIRECTORY = "output/"

def usage():
    """ Print usage message """
    print ""
    print "Usage:"
    print "  $ python -m %s /path/to/container /path/to/output/dir [CONFIG_STRING] [options]" % NAME
    print ""
    print "Options:"
    print "  -v               : verbose output"
    print "  --skip-validator : do not validate"
    print ""
    print "Example:"
    print "  $ python -m %s %s %s" % (NAME, INPUT_FILE, OUTPUT_DIRECTORY)
    print ""
    sys.exit(2)

def main():
    """ Entry point """
    if len(sys.argv) < 3:
        usage()
    container_path = sys.argv[1]
    output_dir = sys.argv[2]
    config_string = None
    verbose = False
    validate = True
    for i in range(3, len(sys.argv)):
        arg = sys.argv[i]
        if arg == "-v":
            verbose = True
        elif arg == "--skip-validator":
            validate = False
        elif (i == 3):
            config_string = arg

    if not gf.can_run_c_extension():
        print "[WARN] Unable to load Python C Extensions"
        print "[WARN] Running the slower pure Python code"
        print "[WARN] See the README file for directions to compile the Python C Extensions"

    logger = Logger(tee=verbose)

    if validate:
        try:
            print "[INFO] Validating the container (specify --skip-validator to bypass)..."
            validator = Validator()
            if config_string is None:
                result = validator.check_container(container_path)
            else:
                result = validator.check_container_from_wizard(container_path, config_string)
            if not result.passed:
                print "[ERRO] The given container is not valid:"
                print result.pretty_print()
                sys.exit(1)
            print "[INFO] Validating the container... done"
        except Exception as exc:
            print "[ERRO] The following error occurred while validating the container:"
            print "[ERRO] %s" % str(exc)
            sys.exit(1)

    try:
        print "[INFO] Loading job from container..."
        executor = ExecuteJob(logger=logger)
        executor.load_job_from_container(container_path, config_string)
        print "[INFO] Loading job from container... done"
    except Exception as exc:
        print "[ERRO] The following error occurred while loading the job:"
        print "[ERRO] %s" % str(exc)
        sys.exit(1)

    try:
        print "[INFO] Executing..."
        executor.execute()
        print "[INFO] Executing... done"
    except Exception as exc:
        print "[ERRO] The following error occurred while executing the job:"
        print "[ERRO] %s" % str(exc)
        sys.exit(1)

    try:
        print "[INFO] Creating output container..."
        path = executor.write_output_container(output_dir)
        print "[INFO] Creating output container... done"
        print "[INFO] Created %s" % path
        executor.clean(True)
    except Exception as exc:
        print "[ERRO] The following error occurred while writing the output container:"
        print "[ERRO] %s" % str(exc)
        sys.exit(1)

    sys.exit(0)

if __name__ == '__main__':
    main()



