#!/usr/bin/env python
# coding=utf-8

"""
Execute a job, passed as a container or
as a container + configuration string (wizard case).
"""

import sys

import aeneas.globalfunctions as gf
from aeneas.executejob import ExecuteJob
from aeneas.logger import Logger
from aeneas.tools import get_rel_path

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
    Copyright 2015,      Alberto Pettarin (www.albertopettarin.it)
    """
__license__ = "GNU AGPL 3"
__version__ = "1.2.0"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

def usage():
    """ Print usage message """
    name = "aeneas.tools.execute_job"
    file_path = get_rel_path("../tests/res/container/job.zip")
    print ""
    print "Usage:"
    print "  $ python -m %s /path/to/container [config_string] /path/to/output/dir [-v]" % name
    print ""
    print "Example:"
    print "  $ python -m %s %s /tmp/" % (name, file_path)
    print ""

def main():
    """ Entry point """
    if len(sys.argv) < 3:
        usage()
        return
    container_path = sys.argv[1]
    config_string = None
    verbose = (sys.argv[-1] == "-v")
    number_of_arguments = 4
    if verbose:
        number_of_arguments += 1
    if len(sys.argv) >= number_of_arguments:
        config_string = sys.argv[2]
        output_dir = sys.argv[3]
    else:
        output_dir = sys.argv[2]

    logger = Logger(tee=verbose)
    executor = ExecuteJob(logger=logger)

    if not gf.can_run_c_extension():
        print "[WARN] Unable to load Python C Extensions"
        print "[WARN] Running the slower pure Python code"
        print "[WARN] See the README file for directions to compile the Python C Extensions"

    print "[INFO] Loading job from container..."
    result = executor.load_job_from_container(container_path, config_string)
    print "[INFO] Loading job from container... done"
    if not result:
        print "[ERRO] The job cannot be loaded from the specified container"
        return

    print "[INFO] Executing..."
    result = executor.execute()
    print "[INFO] Executing... done"

    if not result:
        print "[ERRO] An error occurred while executing the job"
        return

    print "[INFO] Creating output container..."
    result, path = executor.write_output_container(output_dir)
    print "[INFO] Creating output container... done"

    if result:
        print "[INFO] Created %s" % path
    else:
        print "[ERRO] An error occurred while writing the output container"

    executor.clean(True)

if __name__ == '__main__':
    main()



