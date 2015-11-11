#!/usr/bin/env python
# coding=utf-8

"""
Perform validation in one of the following modes:

1. a container
2. a job configuration string
3. a task configuration string
4. a container + configuration string from wizard
5. a job TXT configuration file
6. a job XML configuration file
"""

import codecs
import sys

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

NAME = "aeneas.tools.validate"

CONFIG_FILE = gf.get_rel_path("res/config.txt")
CONTAINER_FILE = gf.get_rel_path("res/job.zip")
JOB_CONFIG_STRING = "job_language=it|os_job_file_name=output.zip|os_job_file_container=zip|is_hierarchy_type=flat"
TASK_CONFIG_STRING = "task_language=it|is_text_type=plain|os_task_file_name=output.txt|os_task_file_format=txt"
WRONG_CONFIG_STRING = "job_language=it|invalid=string"

def usage():
    """ Print usage message """
    print ""
    print "Usage:"
    print "  $ python -m %s config     /path/to/config.txt              [options]" % NAME
    print "  $ python -m %s config     /path/to/config.xml              [options]" % NAME
    print "  $ python -m %s container  /path/to/container               [options]" % NAME
    print "  $ python -m %s [job|task] CONFIG_STRING                    [options]" % NAME
    print "  $ python -m %s wizard     CONFIG_STRING /path/to/container [options]" % NAME
    print ""
    print "Options:"
    print "  -e : return exit code on validation result (zero: passed, nonzero: not passed)"
    print "  -v : verbose output"
    print ""
    print "Examples:"
    print "  $ python -m %s config    %s" % (NAME, CONFIG_FILE)
    print "  $ python -m %s container %s" % (NAME, CONTAINER_FILE)
    print "  $ python -m %s job       \"%s\"" % (NAME, JOB_CONFIG_STRING)
    print "  $ python -m %s task      \"%s\"" % (NAME, TASK_CONFIG_STRING)
    print "  $ python -m %s wizard    \"%s\" %s" % (NAME, WRONG_CONFIG_STRING, CONTAINER_FILE)
    print ""
    sys.exit(2)

def read_file(input_file_path):
    try:
        input_file = None
        input_file = codecs.open(input_file_path, "r", "utf-8")
        contents = input_file.read()
    except:
        print "[ERRO] Unable to read file '%s'" % input_file_path
        sys.exit(1)
    finally:
        input_file.close()
    return contents.encode("utf-8")

def main():
    """ Entry point """
    if len(sys.argv) < 3:
        usage()
    mode = sys.argv[1]
    result = None
    msg = ""
    verbose = False
    exit_code = False
    for i in range(3, len(sys.argv)):
        arg = sys.argv[i]
        if arg == "-e":
            exit_code = True
        if arg == "-v":
            verbose = True

    logger = Logger(tee=verbose)
    validator = Validator(logger=logger)

    if mode == "config":
        if sys.argv[2].endswith(".txt"):
            config_contents = read_file(sys.argv[2])
            result = validator.check_contents_txt_config_file(config_contents, True)
            msg = "TXT configuration"
        elif sys.argv[2].endswith(".xml"):
            config_contents = read_file(sys.argv[2])
            result = validator.check_contents_xml_config_file(config_contents)
            msg = "XML configuration"
        else:
            usage()
    elif mode == "container":
        result = validator.check_container(sys.argv[2])
        msg = "container"
    elif mode == "job":
        result = validator.check_job_configuration(sys.argv[2])
        msg = "job configuration string"
    elif mode == "task":
        result = validator.check_task_configuration(sys.argv[2], external_name=True)
        msg = "task configuration string"
    elif mode == "wizard":
        if len(sys.argv) < 4:
            usage()
        result = validator.check_container_from_wizard(sys.argv[3], sys.argv[2])
        msg = "container + configuration string from wizard"
    else:
        usage()

    if result.passed:
        print "[INFO] Valid %s" % msg
        if len(result.warnings) > 0:
            for warning in result.warnings:
                print "[WARN] " + warning
        if exit_code:
            sys.exit(0)
    else:
        print "[INFO] Invalid %s" % msg
        for error in result.errors:
            print "[ERRO] " + error
        if exit_code:
            sys.exit(4)
    sys.exit(0)

if __name__ == '__main__':
    main()



