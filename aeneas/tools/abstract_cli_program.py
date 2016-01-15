#!/usr/bin/env python
# coding=utf-8

"""
An "abstract" class containing functions common
to the CLI programs in aeneas.tools.
"""

from __future__ import absolute_import
from __future__ import print_function
import io
import os
import sys

from aeneas.logger import Logger
from aeneas.textfile import TextFile
from aeneas.textfile import TextFileFormat
import aeneas.globalfunctions as gf

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

class AbstractCLIProgram(object):
    """
    This class is an "abstract" CLI program.

    To create a new CLI program, create a new class,
    derived from this one, and overload
    ``NAME``, ``HELP``, and ``perform_command()``.

    :param use_sys: if ``True``, call sys.exit
    :type  use_sys: bool
    """

    NAME = gf.file_name_without_extension(__file__)

    AENEAS_URL = u"http://www.readbeyond.it/aeneas/"
    DOCS_URL = u"http://www.readbeyond.it/aeneas/docs/"
    GITHUB_URL = u"https://github.com/ReadBeyond/aeneas/"
    ISSUES_URL = u"https://github.com/ReadBeyond/aeneas/issues/"
    RB_URL = u"http://www.readbeyond.it"

    NO_ERROR_EXIT_CODE = 0
    ERROR_EXIT_CODE = 1
    HELP_EXIT_CODE = 2

    HELP = {
        "description": u"An abstract CLI program",
        "synopsis": [
        ],
        "options": [
        ],
        "parameters": [
        ],
        "examples": [
        ]
    }

    TAG = u"CLI"

    def __init__(self, use_sys=True):
        self.use_sys = use_sys
        self.formal_arguments_raw = []
        self.formal_arguments = []
        self.actual_arguments = []
        self.logger = Logger()
        self.save_log_to_file = False
        self.verbose = False
        self.very_verbose = False

    def _log(self, message, severity=Logger.DEBUG):
        """ Log """
        self.logger.log(message, severity, self.TAG)

    def print_generic(self, msg, prefix=None):
        """
        Print a message and log it.

        :param msg: the message
        :type  msg: Unicode string
        :param prefix: the (optional) prefix
        :type  prefix: Unicode string
        """
        if prefix is None:
            self._log(msg, Logger.INFO)
        if self.use_sys:
            if prefix is not None:
                msg = u"%s %s" % (prefix, msg)
            gf.safe_print(msg)

    def print_error(self, msg):
        """
        Print an error message and log it.

        :param msg: the message
        :type  msg: Unicode string
        """
        self._log(msg, Logger.CRITICAL)
        self.print_generic(msg, u"[ERRO]")

    def print_info(self, msg):
        """
        Print an info message and log it.

        :param msg: the message
        :type  msg: Unicode string
        """
        self._log(msg, Logger.INFO)
        self.print_generic(msg, u"[INFO]")

    def print_warning(self, msg):
        """
        Print a warning message and log it.

        :param msg: the message
        :type  msg: Unicode string
        """
        self._log(msg, Logger.WARNING)
        self.print_generic(msg, u"[WARN]")

    def exit(self, code):
        """
        Exit with the given exit code,
        possibly with ``sys.exit()``.

        :param code: the exit code
        :type  code: int
        :rtype: int
        """
        if self.use_sys:
            sys.exit(code)
        return code

    def print_help(self, short=False):
        """
        Print help message and exit.

        :param short: print short help only
        :type  short: bool
        """
        header = [
            u"",
            u"NAME",
            u"  %s - %s" % (self.NAME, self.HELP["description"]),
            u"",
        ]

        synopsis = [
            u"SYNOPSIS",
            u"  python -m aeneas.tools.%s [-h|--help|--version]" % (self.NAME)
        ]
        if "synopsis" in self.HELP:
            for syn in self.HELP["synopsis"]:
                synopsis.append(u"  python -m aeneas.tools.%s %s [OPTIONS]" % (self.NAME, syn))
        synopsis.append(u"")

        options = [
            u"  -h : print short help and exit",
            u"  --help : print full help and exit",
            u"  --version : print the program name and version and exit",
            u"  -l, --log : log verbose output to tmp file",
            u"  -v, --verbose : verbose output",
            u"  -vv, --very-verbose : verbose output, print date/time values",
        ]
        if "options" in self.HELP:
            for opt in self.HELP["options"]:
                options.append(u"  %s" % (opt))
        options = [u"OPTIONS"] + sorted(options) + [u""]

        parameters = []
        if ("parameters" in self.HELP) and (len(self.HELP["parameters"]) > 0):
            parameters.append(u"PARAMETERS")
            for par in self.HELP["parameters"]:
                parameters.append(u"  %s" % (par))
            parameters.append(u"")

        examples = []
        if ("examples" in self.HELP) and (len(self.HELP["examples"]) > 0):
            examples.append(u"EXAMPLES")
            for exa in self.HELP["examples"]:
                examples.append(u"  python -m aeneas.tools.%s %s" % (self.NAME, exa))
            examples.append(u"")

        footer = [
            u"EXIT CODES",
            u"  %d : no error" % (self.NO_ERROR_EXIT_CODE),
            u"  %d : error" % (self.ERROR_EXIT_CODE),
            u"  %d : help shown, no command run" % (self.HELP_EXIT_CODE),
            u"",
            u"AUTHOR",
            u"  Alberto Pettarin, http://www.albertopettarin.it/",
            u"",
            u"REPORTING BUGS",
            u"  Please use the GitHub Issues Web page : %s" % (self.ISSUES_URL),
            u"",
            u"COPYRIGHT",
            u"  2012-2016, Alberto Pettarin and ReadBeyond Srl",
            u"  This software is available under the terms of the GNU Affero General Public License Version 3",
            u"",
            u"SEE ALSO",
            u"  Code repository  : %s" % (self.GITHUB_URL),
            u"  Documentation    : %s" % (self.DOCS_URL),
            u"  Project Web page : %s" % (self.AENEAS_URL),
            u"",
        ]

        msg = header + synopsis + options + parameters + examples
        if not short:
            msg += footer
        if self.use_sys:
            self.print_generic(u"\n".join(msg))
        return self.exit(self.HELP_EXIT_CODE)

    def print_name_version(self):
        """
        Print program name and version and exit.

        :rtype: int
        """
        if self.use_sys:
            self.print_generic(u"%s v%s" % (self.NAME, __version__))
        return self.exit(self.HELP_EXIT_CODE)

    def run(self, arguments):
        """
        Program entry point.

        Please note that the first item in ``arguments`` is discarded,
        as it is assumed to be the script/invocation name;
        pass a "dumb" placeholder if you call this method with
        an argument different that ``sys.argv``.

        :param arguments: the list of arguments
        :type  arguments: list
        :rtype: int
        """
        # convert arguments into Unicode strings
        if self.use_sys:
            # check that sys.stdin.encoding and sys.stdout.encoding are set to utf-8
            if sys.stdin.encoding not in ["UTF-8", "UTF8"]:
                self.print_warning(u"The default input encoding is not UTF-8.")
                self.print_warning(u"You might want to set 'PYTHONIOENCODING=UTF-8' in your shell.")
            if sys.stdout.encoding not in ["UTF-8", "UTF8"]:
                self.print_warning(u"The default output encoding is not UTF-8.")
                self.print_warning(u"You might want to set 'PYTHONIOENCODING=UTF-8' in your shell.")
            # decode using sys.stdin.encoding
            args = [gf.safe_unicode_stdin(arg) for arg in arguments]
        else:
            # decode using utf-8 (but you should pass Unicode strings as parameters anyway)
            args = [gf.safe_unicode(arg) for arg in arguments]

        if u"-h" in args:
            return self.print_help(short=True)

        if u"--help" in args:
            return self.print_help(short=False)

        if u"--version" in args:
            return self.print_name_version()

        # store formal arguments
        self.formal_arguments_raw = arguments
        self.formal_arguments = args

        # to obtain the actual arguments,
        # remove the first one and "special" switches
        args = args[1:]
        set_args = set(args)

        for flag in set([u"-v", u"--verbose"]) & set_args:
            self.verbose = True
            args.remove(flag)

        for flag in set([u"-vv", u"--very-verbose"]) & set_args:
            self.verbose = True
            self.very_verbose = True
            args.remove(flag)

        for flag in set([u"-l", u"--log"]) & set_args:
            self.save_log_to_file = True
            args.remove(flag)

        if len(args) < 1:
            return self.print_help()

        # store actual arguments
        self.actual_arguments = args

        self.logger = Logger(tee=self.verbose, tee_show_datetime=self.very_verbose)
        self._log([u"Formal arguments: %s", self.formal_arguments])
        self._log([u"Actual arguments: %s", self.actual_arguments])
        exit_code = self.perform_command()
        self._log([u"Execution completed with code %d", exit_code])
        if self.save_log_to_file:
            self._log(u"User requested saving log to file")
            handler, path = gf.tmp_file(u".log")
            self._log([u"Writing log to file '%s'", path])
            with io.open(path, "w", encoding="utf-8") as log_file:
                log_file.write(self.logger.pretty_print())
            if self.use_sys:
                self.print_info(u"Log written to file '%s'" % path)
        return self.exit(exit_code)

    def has_option(self, target):
        """
        Return ``True`` if the actual arguments include
        the specified ``target`` option or,
        if ``target`` is a list of options,
        at least one of them.

        :param target: the option or a list of options
        :type  target: Unicode string or list of Unicode strings
        :rtype: bool
        """
        if isinstance(target, list):
            target_set = set(target)
        else:
            target_set = set([target])
        return len(target_set & set(self.actual_arguments)) > 0

    def has_option_with_value(self, prefix):
        """
        Check if the actual arguments include an option
        starting with the given ``prefix`` and having a value,
        e.g. ``--format=ogg`` for ``prefix="--format"``.

        :param prefix: the option prefix
        :type  prefix: Unicode string
        :rtype: Unicode string or None
        """
        for arg in [arg for arg in self.actual_arguments if arg.startswith(prefix)]:
            lis = arg.split(u"=")
            if len(lis) == 2:
                return lis[1]
        return None

    def perform_command(self):
        """
        Perform command and return the appropriate exit code.

        :rtype: int
        """
        self._log(u"This function should be overloaded in derived classes")
        self._log([u"Invoked with %s", self.actual_arguments])
        return self.NO_ERROR_EXIT_CODE

    def check_c_extensions(self, name=None):
        """
        If C extensions cannot be run, emit a warning
        and return ``False``. Otherwise return ``True``.
        If ``name`` is not ``None``, check just
        the C extension with that name.

        :param name: the name of the Python C extension to test
        :type  name: string
        :rtype: bool
        """
        if not gf.can_run_c_extension(name=name):
            if name is None:
                self.print_warning(u"Unable to load Python C Extensions")
            else:
                self.print_warning(u"Unable to load Python C Extension %s" % (name))
            self.print_warning(u"Running the slower pure Python code")
            self.print_warning(u"See the documentation for directions to compile the Python C Extensions")
            return False
        return True

    def check_input_file(self, path):
        """
        If the given path does not exist, emit an error
        and return ``False``. Otherwise return ``True``.

        :param path: the path of the input file
        :type  path: string (path)
        :rtype: bool
        """
        if not gf.file_can_be_read(path):
            self.print_error(u"Unable to read file '%s'" % (path))
            self.print_error(u"Make sure the file path is written/escaped correctly and that you have read permission on it")
            return False
        return True

    def check_output_file(self, path):
        """
        If the given path cannot be written, emit an error
        and return ``False``. Otherwise return ``True``.

        :param path: the path of the output file
        :type  path: string (path)
        :rtype: bool
        """
        if not gf.file_can_be_written(path):
            self.print_error(u"Unable to create file '%s'" % (path))
            self.print_error(u"Make sure the file path is written/escaped correctly and that you have write permission on it")
            return False
        return True

    def check_output_directory(self, path):
        """
        If the given directory cannot be written, emit an error
        and return ``False``. Otherwise return ``True``.

        :param path: the path of the output directory
        :type  path: string (path)
        :rtype: bool
        """
        if not os.path.isdir(path):
            self.print_error(u"Directory '%s' does not exist" % (path))
            return False
        test_file = os.path.join(path, u"file.test")
        if not gf.file_can_be_written(test_file):
            self.print_error(u"Unable to write inside directory '%s'" % (path))
            self.print_error(u"Make sure the directory path is written/escaped correctly and that you have write permission on it")
            return False
        return True

    def get_text_file(self, text_format, text, parameters):
        if text_format == u"list":
            text_file = TextFile(logger=self.logger)
            text_file.read_from_list(text.split(u"|"))
            return text_file
        else:
            if text_format not in TextFileFormat.ALLOWED_VALUES:
                self.print_error(u"File format '%s' is not allowed" % (text_format))
                self.print_error(u"Allowed text file formats: %s" % (" ".join(TextFileFormat.ALLOWED_VALUES)))
                return None
            try:
                return TextFile(text, text_format, parameters, logger=self.logger)
            except OSError:
                self.print_error(u"Cannot read file '%s'" % (text))
            return None



def main():
    """
    Execute program.
    """
    AbstractCLIProgram().run(arguments=sys.argv)

if __name__ == '__main__':
    main()



