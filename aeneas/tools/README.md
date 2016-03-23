# aeneas Tools 

This Python module (directory) contains the command line (CLI) tools for ``aeneas``.

Each tool, contained in a separate Python source file,
extends the abstract class ``AbstractCLIProgram``
defined in ``abstract_cli_program.py``.

``AbstractCLIProgram`` provides the following main functions, common to all CLI tools:

* ``run()`` runs the tool: first, it does some basic command line handling and then calls ``perform_command()``;
* ``perform_command()`` provides the actual logic of the tool and must be implemented by the concrete subclasses; and
* ``print_help()`` and ``print_name_version()`` print the usage and version messages.

In the overridden ``perform_command()`` function you might want to use the following utility functions,
all defined in ``AbstractCLIProgram``:

* ``print_generic()``, ``print_error()``, ``print_info()``, ``print_success()``, and ``print_warning()`` to print messages to stdout;
* ``exit()`` to terminate the execution of your tool, e.g. on critical errors, returning a suitable exit code;
* ``has_option()`` and ``has_option_with_value()`` to check for options in the arguments passed by the caller;
* ``check_c_extensions()`` to check for the availability of Python C extensions;
* ``check_input_file()`` to check that the given input path exists and it is readable;
* ``check_output_file()`` and ``check_output_directory()`` to check that the given output path can be written;
* ``get_text_file()`` to create a ``aeneas.TextFile`` object from the given arguments.

In particular, be sure to use the ``print_*`` functions listed above,
instead of calling the default Python ``print()`` function,
for two reasons:

* the above calls implicitly call ``log()``, logging all messages;
* the above calls do not output to sys.stdout if ``use_sys = False`` (see below).


## Running a tool on the command line

When run on the command line, each tool instantiates the corresponding class
and calls the ``run()`` function, passing the ``sys.argv`` list as its only argument.

The ``run()`` function, after doing some basic command line handling,
calls the overridden ``perform_command()`` function,
which contains the logic specific of each tool.

For example, ``execute_task.py`` creates an ``ExecuteTaskCLI`` object,
which executes the given **aeneas** Task, that is, it outputs a synchronization map
for the **aeneas** Task, as specified in the input arguments:

```python
...

class ExecuteTaskCLI(AbstractCLIProgram):
    ...

    def perform_command(self):
        <ACTUAL LOGIC FOR EXECUTING AN AENEAS TASK>

def main():
    """
    Execute program.
    """
    ExecuteTaskCLI().run(arguments=sys.argv)

if __name__ == '__main__':
    main()
```

Note that a suitable exit code is returned to the shell.


## Running a tool from other Python code

You can also import and run each tool from other Python code.

For example:

```python
from aeneas.tools.execute_task import ExecuteTaskCLI

args = [
    "dummy",
    "input/audio.mp3",
    "input/plain.txt",
    "task_language=en|is_text_type=plain|os_task_file_format=json",
    "output/sonnet.json"
]
exit_code = ExecuteTaskCLI(use_sys=False).run(arguments=args)
```

will compute a JSON sync map file ``sonnet.json`` for the Task
whose audio and text files are ``audio.mp3`` and ``plain.txt``
(the latter in ``plain`` format), assuming English (``en``) as its language.

In the constructor of ``ExecuteTaskCLI`` you might want to specify
``use_sys=False`` so that the tool never calls a ``sys.exit()`` or
outputs to stdout directly.

Note that the first element of the ``arguments`` list must be a dummy placeholder,
to formally resemble the ``sys.argv[0]``.
(This defect will be removed in a future version.)

You can tell whether the ``run()`` call succeeded by inspecting the returned ``exit_code``.



