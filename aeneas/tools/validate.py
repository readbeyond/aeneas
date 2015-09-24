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

import sys

from aeneas.validator import Validator
from aeneas.tools import get_rel_path

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
    Copyright 2015,      Alberto Pettarin (www.albertopettarin.it)
    """
__license__ = "GNU AGPL 3"
__version__ = "1.1.2"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

def usage():
    """ Print usage message """
    name = "aeneas.tools.validate"
    file_path_1 = get_rel_path("../tests/res/container/job.zip")
    file_path_2 = get_rel_path("../tests/res/container/job/config.txt")
    print ""
    print "Usage:"
    print "  $ python -m %s config /path/to/config.txt" % name
    print "  $ python -m %s config /path/to/config.xml" % name
    print "  $ python -m %s container /path/to/container" % name
    print "  $ python -m %s [job|task] config_string" % name
    print "  $ python -m %s wizard /path/to/container config_string" % name
    print ""
    print "Example:"
    print "  $ python -m %s config %s" % (name, file_path_2)
    print "  $ python -m %s container %s" % (name, file_path_1)
    print "  $ python -m %s job \"job_language=it|os_job_file_name=output.zip|os_job_file_container=zip|is_hierarchy_type=flat\"" % name
    print "  $ python -m %s task \"task_language=it|is_text_type=plain|os_task_file_name=output.txt|os_task_file_format=txt\"" % name
    print "  $ python -m %s wizard %s \"job_language=it|generate_an_error=1\"" % (name, file_path_2)
    print ""

def main():
    """ Entry point """
    if len(sys.argv) < 3:
        usage()
        return
    validator = Validator()
    mode = sys.argv[1]
    result = None
    msg = ""

    if mode == "config":
        if sys.argv[2].endswith(".txt"):
            try:
                config_file = open(sys.argv[2], "r")
                config_contents = config_file.read()
                config_file.close()
                result = validator.check_contents_txt_config_file(config_contents, True)
                msg = "TXT configuration"
            except:
                print "[ERRO] Unable to read file %s" % sys.argv[2]
        elif sys.argv[2].endswith(".xml"):
            try:
                config_file = open(sys.argv[2], "r")
                config_contents = config_file.read()
                config_file.close()
                result = validator.check_contents_xml_config_file(config_contents)
                msg = "XML configuration"
            except:
                print "[ERRO] Unable to read file %s" % sys.argv[2]
        else:
            usage()
            return
    elif mode == "container":
        result = validator.check_container(sys.argv[2])
        msg = "container"
    elif mode == "job":
        result = validator.check_job_configuration(sys.argv[2])
        msg = "job configuration string"
    elif mode == "task":
        result = validator.check_task_configuration(sys.argv[2])
        msg = "task configuration string"
    elif mode == "wizard":
        if len(sys.argv) < 4:
            usage()
            return
        result = validator.check_container_from_wizard(sys.argv[2], sys.argv[3])
        msg = "container + configuration string from wizard"
    else:
        usage()
        return

    if result.passed:
        print "[INFO] Valid %s" % msg
        if len(result.warnings) > 0:
            for warning in result.warnings:
                print "[WARN] " + warning
    else:
        print "[INFO] Invalid %s" % msg
        for error in result.errors:
            print "[ERRO] " + error

if __name__ == '__main__':
    main()



