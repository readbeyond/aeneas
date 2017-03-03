# Testing

The unit and integration tests are contained in the
[``aeneas.tests``](https://github.com/readbeyond/aeneas/blob/master/aeneas/tests/)
subpackage.


## Unit and Integration

The tests cover both the library ``aeneas``
and the CLI tools ``aeneas.tools``.

Currently, there are more than 1,200 tests.

### Structure of the Tests

Each test file roughly corresponds to the source file being tested,
with the exception of a couple of modules,
whose tests have been split across multiple test files.

The following naming convention is used:

* regular tests are specified in files with prefix ``test_``;
* tests requiring network access have prefix ``net_test_``;
* tests with long running time have prefix ``long_test_``;
* tests included in the benchmark suite have prefix ``bench_test_``;
* tests for ``aeneas.tools`` have prefix ``tool_test_``.

### Running the Tests

The test harness is the standard Python library ``unittest``,
hence no additional testing libraries/frameworks
should be needed in order to run the tests.

Tests are not distributed via PyPI,
since some of them require large audio or data files.
If you installed ``aeneas`` using PyPI (e.g., via ``pip``),
you cannot run the tests.
You must ``git clone`` the main repository instead.

Certain tests might fail
if you do not set your environment up properly,
for example, if you do not have optional libraries/tools installed
(e.g., ``festival`` or ``speect``, etc.).

In the root directory of the repository
there is a ``run_all_unit_tests.py`` script
that helps running the unit/integration tests.
Run:

```bash
$ # get the usage
$ python run_all_unit_tests.py -h

$ # run all regular tests (< 3 minutes)
$ python run_all_unit_tests.py

$ # run all tool tests (< 5 minutes)
$ python run_all_unit_tests.py -t

$ # run all long tests (< 15 minutes)
$ python run_all_unit_tests.py -l

$ # run all benchmark tests (< 15 minutes)
$ python run_all_unit_tests.py -b

$ # run all regular tests with verbose output including error messages
$ python run_all_unit_tests.py -v

$ # run all network tests (requires an Internet connection)
$ python run_all_unit_tests.py -n

$ # run only textfile tests
$ python run_all_unit_tests.py textfile

$ # run only run_vad tool tests
$ python run_all_unit_tests.py -t run_vad
```

The
[``venvs``](https://github.com/readbeyond/aeneas/blob/master/venvs/)
directory contains some scripts to automate
the creation and management of virtual enviroments
for testing purposes.

### Release Process

Before releasing a new version of ``aeneas``,
all tests (regular, tools, long, benchmark, network)
are run on **all**
[supported platforms and Python interpreters](https://github.com/readbeyond/aeneas/blob/master/wiki/PLATFORMS.md).

**Any failure blocks the release process**,
until the failure is removed.

Tests might be run on other (i.e., not officially supported)
platforms or interpreters as well,
but any failure on them does not automatically block the release process.

Please consider **sponsoring** the project
to add more extensive tests
or to include your platform in the release testing process.
See the
[README file](https://github.com/readbeyond/aeneas/README.md)
for details.


## Performance

Performance tests are located in a
[separate repository](https://github.com/readbeyond/aeneas-benchmark),
with the corresponding documentation.

The benchmark results can be consulted online at the following URL:
[https://readbeyond.github.io/aeneas-benchmark/](https://readbeyond.github.io/aeneas-benchmark/)
