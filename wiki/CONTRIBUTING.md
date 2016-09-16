# Code Contribution Guide

If you are able to contribute code directly, that is awesome!

I will be glad to merge it!
Just a few rules, to make life easier for both you and me:

1. Respect the branch policy explained below.

2. Respect the code style guide detailed below.

3. Add unit tests for the code you are submitting,
   either adding them to the existing unit tests
   or creating a new file
   in `aeneas/tests/`.

4. **By opening a pull request,
   you automatically agree to apply
   the AGPL v3 license
   to the code you contribute.**


## Branch Policy

At all times:

* the `master` branch holds the latest stable version of **aeneas**;
* the `devel` branch holds the latest development version.

Once the code on the `devel` branch is ready to be released,
its content will be pushed to the `master` branch,
and to PyPI and other downstream channels.

Do **not** fork and edit the `master` branch.
Instead, fork the `devel` branch and always submit pull requests
to the `devel` branch.

**Pull requests submitted against the `master` branch
will not be merged, and the submitter will be asked
to submit a new one against the `devel` branch.**

This policy guarantees that the code on the `master` branch
is stable and vetted at all times.


## Code Style Guide

The existing code base style follows
[PEP 8](https://www.python.org/dev/peps/pep-0008/),
please make your contribution adhere to it as well.

Install ``pep8`` (``pip install pep8``) and
strictly check any Python source code with it:

```bash
$ pep8 srcfile.py
```

The only exception is that you can ignore error ``E501``
(aka "line longer than 79 characters"):

```bash
$ pep8 --ignore=E501 srcfile.py
```

See also the
[Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
for further details.

