# Tested Platforms 

**aeneas** has been confirmed to work on the following systems:

| OS             | 32/64 bit | Python 2.7 | Python 3.4/3.5  |
|----------------|-----------|------------|-----------------|
| Debian         | 64        | Yes        | Yes             |
| Debian         | 32        | Yes        | Yes             |
| Ubuntu         | 64        | Yes        | Yes             |
| Gentoo         | 64        | Yes        | Unknown         |
| Slackware      | 64        | Yes        | Unknown         |
| Mac OS X 10.9  | 64        | Yes        | Unknown         |
| Mac OS X 10.10 | 64        | Yes        | Unknown         |
| Mac OS X 10.11 | 64        | Yes        | Unknown         |
| Windows Vista  | 32        | Yes (1)    | Yes (1, 2)      |
| Windows 7      | 64        | Yes (1)    | Yes (1, 2)      |
| Windows 8.1    | 64        | Yes (1)    | Unknown (1, 2)  |
| Windows 10     | 64        | Yes (1)    | Yes (1, 2)      |

**Notes**
(1) The ``cew`` Python C extension to speed up text synthesis
is available only on Linux, Mac OS X, and Windows 32 bit at the moment.
(2) On Windows and Python 3.4/3.5, compiling the Python C extensions
is quite complex; however, running **aeneas** in pure Python mode
has been confirmed to work.

Do you need official support for another OS?
Consider **sponsoring** this project!
