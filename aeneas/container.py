#!/usr/bin/env python
# coding=utf-8

"""
A container is an abstraction for a group of files (entries)
compressed into an archive file (e.g., ZIP or TAR)
or uncompressed inside a directory.

This module contains two main classes.

1. :class:`aeneas.container.Container`
   is the main class, exposing functions
   like extracting all or just one entry,
   listing the entries in the container, etc.
2. :class:`aeneas.container.ContainerFormat`
   is an enumeration of the supported container formats.
"""

import os
import tarfile
import zipfile

from aeneas.logger import Logger
import aeneas.globalconstants as gc
import aeneas.globalfunctions as gf

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
    Copyright 2015,      Alberto Pettarin (www.albertopettarin.it)
    """
__license__ = "GNU AGPL v3"
__version__ = "1.3.2"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

class ContainerFormat(object):
    """
    Enumeration of the supported container formats.
    """

    EPUB = "epub"
    """ EPUB container """

    TAR = "tar"
    """ TAR container (without compression) """

    TAR_GZ = "tar.gz"
    """ TAR container with GZ compression"""

    TAR_BZ2 = "tar.bz2"
    """ TAR container with BZ2 compression """

    UNPACKED = "unpacked"
    """ Unpacked container (i.e., a directory) """

    ZIP = "zip"
    """ ZIP container """

    ALLOWED_VALUES = [EPUB, TAR, TAR_GZ, TAR_BZ2, UNPACKED, ZIP]
    """ List of all the allowed values """

class Container(object):
    """
    An abstraction for different archive formats like ZIP or TAR,
    exposing common functions like extracting all files or
    a single file, listing the files, etc.

    An (uncompressed) directory can be used in lieu of a compressed file.

    :param file_path: the path to the container file (or directory)
    :type  file_path: string (path)
    :param container_format: the format of the container
    :type  container_format: :class:`aeneas.container.ContainerFormat`
    :param logger: the logger object
    :type  logger: :class:`aeneas.logger.Logger`

    :raise TypeError: if ``file_path`` is None
    :raise ValueError: if ``container_format`` is not None and is not an allowed value
    """

    TAG = "Container"

    def __init__(self, file_path, container_format=None, logger=None):
        if file_path is None:
            raise TypeError("File path is None")
        if (
                (container_format is not None) and
                (container_format not in ContainerFormat.ALLOWED_VALUES)
        ):
            raise ValueError("Container format not allowed")
        self.file_path = file_path
        self.container_format = container_format
        self.actual_container = None
        self.logger = logger
        if self.logger is None:
            self.logger = Logger()
        self._log("Setting actual Container object")
        self._set_actual_container()

    def _log(self, message, severity=Logger.DEBUG):
        """ Log """
        self.logger.log(message, severity, self.TAG)

    @property
    def file_path(self):
        """
        The path of this container.

        :rtype: string (path)
        """
        return self.__file_path
    @file_path.setter
    def file_path(self, file_path):
        self.__file_path = file_path

    @property
    def container_format(self):
        """
        The format of this container.

        :rtype: :class:`aeneas.container.ContainerFormat`
        """
        return self.__container_format
    @container_format.setter
    def container_format(self, container_format):
        self.__container_format = container_format

    @property
    def has_config_xml(self):
        """
        Return ``True`` if there is an XML config file in this container,
        ``False`` otherwise.

        :rtype: bool

        :raise: see ``entries()``
        """
        return self.entry_config_xml is not None

    @property
    def entry_config_xml(self):
        """
        Return the entry (path inside the container)
        of the XML config file in this container,
        or ``None`` if not present.

        :rtype: string (path)

        :raise: see ``entries()``
        """
        return self.find_entry(gc.CONFIG_XML_FILE_NAME, exact=False)

    @property
    def has_config_txt(self):
        """
        Return ``True`` if there is a TXT config file in this container,
        ``False`` otherwise.

        :rtype: bool

        :raise: see ``entries()``
        """
        return self.entry_config_txt is not None

    @property
    def entry_config_txt(self):
        """
        Return the entry (path inside the container)
        of the TXT config file in this container,
        or ``None`` if not present.

        :rtype: string (path)

        :raise: see ``entries()``
        """
        return self.find_entry(gc.CONFIG_TXT_FILE_NAME, exact=False)

    @property
    def is_safe(self):
        """
        Return ``True`` if the container can be safely extracted,
        that is, if all its entries are safe, ``False`` otherwise.

        :rtype: bool

        :raise: see ``entries()``
        """
        self._log("Checking if this container is safe")
        entries = self.entries()
        for entry in entries:
            if not self.is_entry_safe(entry):
                self._log(["This container is not safe: found unsafe entry '%s'", entry])
                return False
        self._log("This container is safe")
        return True

    def is_entry_safe(self, entry):
        """
        Return ``True`` if ``entry`` can be safely extracted,
        that is, if it does start with ``/`` or ``../``
        after path normalization, ``False`` otherwise.

        :rtype: bool
        """
        normalized = os.path.normpath(entry)
        if normalized.startswith("/") or normalized.startswith("../"):
            self._log(["Entry '%s' is not safe", entry])
            return False
        self._log(["Entry '%s' is safe", entry])
        return True

    def entries(self):
        """
        Return the sorted list of entries in this container,
        each represented by its full path inside the container.

        :rtype: list of strings (path)

        :raise TypeError: if this container does not exist
        :raise IOError: if an error occurred reading the given container (e.g., empty file, damaged file, etc.)
        """
        self._log("Getting entries")
        if not self.exists():
            raise TypeError("This container does not exist (wrong path?)")
        if self.actual_container is None:
            raise TypeError("The actual container object has not been set")
        return self.actual_container.entries()

    def find_entry(self, entry, exact=True):
        """
        Return the full path to the first entry whose file name equals
        the given ``entry`` path.

        Return ``None`` if the entry cannot be found.

        If ``exact`` is ``True``, the path must be exact,
        otherwise the comparison is done only on the file name.

        Example: ::

            entry = "config.txt"

        might match: ::

            config.txt
            foo/config.txt (if exact = False)
            foo/bar/config.txt (if exact = False)

        :param entry: the entry name to be searched for
        :type  entry: string (path)
        :param exact: look for the exact entry path
        :type  exact: bool
        :rtype: string (path)

        :raise: see ``entries()``
        """
        if exact:
            self._log(["Finding entry '%s' with exact=True", entry])
            if entry in self.entries():
                self._log(["Found entry '%s'", entry])
                return entry
        else:
            self._log(["Finding entry '%s' with exact=False", entry])
            for ent in self.entries():
                if os.path.basename(ent) == entry:
                    self._log(["Found entry '%s'", ent])
                    return ent
        self._log(["Entry '%s' not found", entry])
        return None

    def read_entry(self, entry):
        """
        Read the contents of an entry in this container,
        and return them as a string.

        Return ``None`` if the entry is not safe
        or it cannot be found.

        :rtype: string

        :raise: see ``entries()``
        """
        if not self.is_entry_safe(entry):
            self._log(["Accessing entry '%s' is not safe", entry])
            return None

        if not entry in self.entries():
            self._log(["Entry '%s' not found in this container", entry])
            return None

        self._log(["Reading contents of entry '%s'", entry])
        try:
            return self.actual_container.read_entry(entry)
        except:
            self._log(["An error occurred while reading the contents of '%s'", entry])
            return None

    def decompress(self, output_path):
        """
        Decompress the entire container into the given directory.

        :param output_path: path of the destination directory
        :type  output_path: string (path)

        :raise TypeError: if this container does not exist
        :raise ValueError: if this container contains unsafe entries, or ``output_path`` is not an existing directory
        :raise IOError: if an error occurred decompressing the given container (e.g., empty file, damaged file, etc.)
        """
        self._log(["Decompressing the container into '%s'", output_path])
        if not self.exists():
            raise TypeError("This container does not exist (wrong path?)")
        if self.actual_container is None:
            raise TypeError("The actual container object has not been set")
        if not gf.directory_exists(output_path):
            raise ValueError("The output_path is not an existing directory")
        if not self.is_safe:
            raise ValueError("This container contains unsafe entries")
        self.actual_container.decompress(output_path)

    def compress(self, input_path):
        """
        Compress the contents of the given directory.

        :param input_path: path of the input directory
        :type  input_path: string (path)

        :raise TypeError: if the container path has not been set
        :raise ValueError: if ``input_path`` is not an existing directory
        :raise IOError: if an error occurred compressing the given container (e.g., empty file, damaged file, etc.)
        """
        self._log(["Compressing '%s' into this container", input_path])

        if self.file_path == None:
            raise TypeError("The container path has not been set")
        if self.actual_container is None:
            raise TypeError("The actual container object has not been set")
        if not gf.directory_exists(input_path):
            raise ValueError("The input_path is not an existing directory")
        gf.ensure_parent_directory(input_path)
        self.actual_container.compress(input_path)

    def exists(self):
        """
        Return ``True`` if the container has its path set and it exists,
        ``False`` otherwise.

        :rtype: boolean
        """
        return gf.file_exists(self.file_path) or gf.directory_exists(self.file_path)

    def _set_actual_container(self):
        """
        Set the actual container, based on the specified container format.

        If the container format is not specified,
        infer it from the (lowercased) extension of the file path.
        If the format cannot be inferred, it is assumed to be
        of type :class:`aeneas.container.ContainerFormat.UNPACKED`
        (unpacked directory).
        """
        self._log("Setting actual container")

        # infer container format
        if self.container_format is None:
            self._log("Inferring actual container format")
            path_lowercased = self.file_path.lower()
            self._log(["Lowercased file path: '%s'", path_lowercased])
            if path_lowercased.endswith(ContainerFormat.ZIP):
                self.container_format = ContainerFormat.ZIP
            elif path_lowercased.endswith(ContainerFormat.EPUB):
                self.container_format = ContainerFormat.EPUB
            elif path_lowercased.endswith(ContainerFormat.TAR):
                self.container_format = ContainerFormat.TAR
            elif path_lowercased.endswith(ContainerFormat.TAR_GZ):
                self.container_format = ContainerFormat.TAR_GZ
            elif path_lowercased.endswith(ContainerFormat.TAR_BZ2):
                self.container_format = ContainerFormat.TAR_BZ2
            else:
                self.container_format = ContainerFormat.UNPACKED
            self._log(["Inferred format: '%s'", self.container_format])

        # set the actual container
        self._log("Setting actual container")
        if self.container_format == ContainerFormat.ZIP:
            self.actual_container = _ContainerZIP(self.file_path)
        elif self.container_format == ContainerFormat.EPUB:
            self.actual_container = _ContainerZIP(self.file_path)
        elif self.container_format == ContainerFormat.TAR:
            self.actual_container = _ContainerTAR(self.file_path, "")
        elif self.container_format == ContainerFormat.TAR_GZ:
            self.actual_container = _ContainerTAR(self.file_path, ":gz")
        elif self.container_format == ContainerFormat.TAR_BZ2:
            self.actual_container = _ContainerTAR(self.file_path, ":bz2")
        elif self.container_format == ContainerFormat.UNPACKED:
            self.actual_container = _ContainerUnpacked(self.file_path)
        self._log(["Actual container format: '%s'", self.container_format])
        self._log("Actual container set")



class _ContainerTAR(object):
    """
    A TAR container.
    """

    TAG = "ContainerTAR"

    def __init__(self, file_path, variant, logger=None):
        self.file_path = file_path
        self.variant = variant
        self.logger = logger
        if self.logger is None:
            self.logger = Logger()

    def entries(self):
        try:
            argument = "r" + self.variant
            tar_file = tarfile.open(self.file_path, argument)
            result = [e.name for e in tar_file.getmembers() if e.isfile()]
            tar_file.close()
            return sorted(result)
        except:
            raise IOError("Cannot read entries from TAR file")

    def read_entry(self, entry):
        try:
            argument = "r" + self.variant
            tar_file = tarfile.open(self.file_path, argument)
            tar_entry = tar_file.extractfile(entry)
            result = tar_entry.read()
            tar_entry.close()
            tar_file.close()
            return result
        except:
            raise IOError("Cannot read entry from TAR file")

    def decompress(self, output_path):
        try:
            argument = "r" + self.variant
            tar_file = tarfile.open(self.file_path, argument)
            tar_file.extractall(output_path)
            tar_file.close()
        except:
            raise IOError("Cannot decompress TAR file")

    def compress(self, input_path):
        try:
            argument = "w" + self.variant
            tar_file = tarfile.open(self.file_path, argument)
            root_len = len(os.path.abspath(input_path))
            for root, dirs, files in os.walk(input_path):
                archive_root = os.path.abspath(root)[root_len:]
                for f in files:
                    fullpath = os.path.join(root, f)
                    archive_name = os.path.join(archive_root, f)
                    tar_file.add(name=fullpath, arcname=archive_name)
            tar_file.close()
        except:
            raise IOError("Cannot compress TAR File")

class _ContainerZIP(object):
    """
    A ZIP container.
    """

    TAG = "ContainerZIP"

    def __init__(self, file_path, logger=None):
        self.file_path = file_path
        self.logger = logger
        if self.logger is None:
            self.logger = Logger()

    def entries(self):
        try:
            zip_file = zipfile.ZipFile(self.file_path)
            result = [e for e in zip_file.namelist() if not e.endswith("/")]
            zip_file.close()
            return sorted(result)
        except:
            raise IOError("Cannot read entries from ZIP file")

    def read_entry(self, entry):
        try:
            zip_file = zipfile.ZipFile(self.file_path)
            zip_entry = zip_file.open(entry)
            result = zip_entry.read()
            zip_entry.close()
            zip_file.close()
            return result
        except:
            raise IOError("Cannot read entry from ZIP file")

    def decompress(self, output_path):
        try:
            zip_file = zipfile.ZipFile(self.file_path)
            zip_file.extractall(output_path)
            zip_file.close()
        except:
            raise IOError("Cannot decompress ZIP file")

    def compress(self, input_path):
        try:
            zip_file = zipfile.ZipFile(self.file_path, "w")
            root_len = len(os.path.abspath(input_path))
            for root, dirs, files in os.walk(input_path):
                archive_root = os.path.abspath(root)[root_len:]
                for f in files:
                    fullpath = os.path.join(root, f)
                    archive_name = os.path.join(archive_root, f)
                    zip_file.write(fullpath, archive_name)
            zip_file.close()
        except:
            raise IOError("Cannot compress ZIP file")

class _ContainerUnpacked(object):
    """
    An unpacked container.
    """

    TAG = "ContainerUnpacked"

    def __init__(self, file_path, logger=None):
        self.file_path = file_path
        self.logger = logger
        if self.logger is None:
            self.logger = Logger()

    def entries(self):
        try:
            result = []
            root_len = len(os.path.abspath(self.file_path))
            for current_dir, dirs, files in os.walk(self.file_path):
                current_dir_abs = os.path.abspath(current_dir)
                for f in files:
                    relative_path = os.path.join(current_dir_abs, f)[root_len+1:]
                    result.append(relative_path)
            return sorted(result)
        except:
            raise IOError("Cannot read entries from unpacked")

    def read_entry(self, entry):
        try:
            unpacked_entry = file(os.path.join(self.file_path, entry), "r")
            result = unpacked_entry.read()
            unpacked_entry.close()
            return result
        except:
            raise IOError("Cannot read entry from unpacked")

    def decompress(self, output_path):
        try:
            if os.path.abspath(output_path) == os.path.abspath(self.file_path):
                return
            gf.copytree(self.file_path, output_path)
        except:
            raise IOError("Cannot decompress unpacked")

    def compress(self, input_path):
        try:
            if os.path.abspath(input_path) == os.path.abspath(self.file_path):
                return
            gf.copytree(input_path, self.file_path)
        except:
            raise IOError("Cannot compress unpacked")


