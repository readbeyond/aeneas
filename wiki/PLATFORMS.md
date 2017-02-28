# Tested Platforms

**aeneas** has been confirmed to work on the following systems:

| OS                | 32/64 bit | Python 2.7 | Python 3.5 | PyPy 5.x |
|-------------------|-----------|------------|------------|----------|
| Arch              | 64        | Yes        | Yes        | Unknown  |
| Debian            | 32        | Yes        | Yes        | Unknown  |
| Debian            | 64        | Yes        | Yes        | Yes      |
| Ubuntu            | 64        | Yes        | Yes        | Yes      |
| Gentoo            | 64        | Yes        | Unknown    | Unknown  |
| Slackware         | 64        | Yes        | Unknown    | Unknown  |
| Mac OS X 10.7 (3) | 64        | Yes        | Yes        | Unknown  |
| Mac OS X 10.8 (3) | 64        | Yes        | Yes        | Unknown  |
| Mac OS X 10.9     | 64        | Yes        | Yes        | Unknown  |
| Mac OS X 10.10    | 64        | Yes        | Yes        | Unknown  |
| Mac OS X 10.11    | 64        | Yes        | Yes        | Unknown  |
| Mac OS X 10.12    | 64        | Yes        | Yes        | Yes      |
| Windows Vista     | 32        | Yes (1)    | Yes (1, 2) | Unknown  |
| Windows 7         | 32        | Yes (1)    | Yes (1, 2) | Unknown  |
| Windows 7         | 64        | Yes (1)    | Yes (1, 2) | Unknown  |
| Windows 8.1       | 32        | Yes (1)    | Yes (1, 2) | Unknown  |
| Windows 8.1       | 64        | Yes (1)    | Yes (1, 2) | Unknown  |
| Windows 10        | 64        | Yes (1)    | Yes (1, 2) | Unknown  |

**Notes**

(1) The ``cew`` Python C extension to speed up text synthesis
with eSpeak is available on Linux, Mac OS X,
and Windows 32 bit at the moment.

(2) On Windows and Python 3.5, compiling the Python C extensions
is quite complex; however, running **aeneas** in pure Python mode
has been confirmed to work.

(3) No longer supported by Apple, and hence no longer tested.
Some previous versions of ``aeneas`` worked on them.
As long as the OS/Python dependencies are available,
newer versions of ``aeneas`` are expected to work too.

(*) The ``cfw`` Python C++ extension to speed up text synthesis
with Festival is experimental and it only works on Linux and Mac OS X.

Do you need official support for another OS?
Consider **sponsoring** this project!
See the
[README file](https://github.com/readbeyond/aeneas/README.md)
for details.
