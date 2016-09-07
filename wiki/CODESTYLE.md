# Code Style Guide

The existing code base style follows [PEP8](https://www.python.org/dev/peps/pep-0008/).

Install ``pep8`` (``pip install pep8``) and check any Python source code with it:

```bash
$ pep8 srcfile.py
```

Optionally, you can disable ``E501``, aka "line longer than 79 characters":

```bash
$ pep8 --ignore=E501 srcfile.py
```

See also the
[Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
for further details.

