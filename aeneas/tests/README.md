# aeneas Tests 

This Python module (directory) contains the ``unittest`` tests for ``aeneas``.

The tests are (roughly) divided by source file in the main ``aeneas`` library.

The following naming convention is used:

* regular tests are specified in files with prefix ``test_``;
* tests requiring network access have prefix ``net_test_``;
* tests with long running time have prefix ``long_test_``;
* tests for ``aeneas.tools`` have prefix ``tool_test_``.

Use the ``run_all_unit_tests.py`` in the git root directory to run the tests
with fine control and with different Python interpreters.



