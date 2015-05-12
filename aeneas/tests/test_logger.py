#!/usr/bin/env python
# coding=utf-8

import os
import sys
import tempfile
import unittest

from aeneas.logger import Logger

class TestLogger(unittest.TestCase):

    def test_log(self):
        logger = Logger(tee=False, indentation=4)
        logger.log("Message 1", Logger.DEBUG)
        logger.log("Message 2", Logger.INFO)
        logger.log("Message 3", Logger.WARNING)
        logger.log("Message 4", Logger.CRITICAL)
        self.assertEqual(len(logger), 4)

    def test_clear(self):
        logger = Logger(tee=False, indentation=4)
        logger.log("Message 1", Logger.DEBUG)
        logger.log("Message 2", Logger.INFO)
        logger.log("Message 3", Logger.WARNING)
        logger.log("Message 4", Logger.CRITICAL)
        self.assertEqual(len(logger), 4)
        logger.clear()
        self.assertEqual(len(logger), 0)

    def test_change_indentation(self):
        logger = Logger(tee=False, indentation=4)
        self.assertEqual(logger.indentation, 4)
        logger.log("Message 1", Logger.DEBUG)
        logger.log("Message 2", Logger.INFO)
        logger.indentation = 2
        self.assertEqual(logger.indentation, 2)
        logger.log("Message 3", Logger.WARNING)
        logger.log("Message 4", Logger.CRITICAL)
        logger.indentation = 0
        self.assertEqual(logger.indentation, 0)

    def test_tag(self):
        logger = Logger(tee=False, indentation=4)
        logger.log("Message 1", Logger.DEBUG, tag="TEST")
        logger.log("Message 2", Logger.DEBUG)
        logger.log("Message 3", Logger.DEBUG, tag="TEST")
        logger.log("Message 4", Logger.DEBUG)
        strings = logger.to_list_of_strings()
        self.assertEqual(strings[0].find("TEST") > -1, True)
        self.assertEqual(strings[1].find("TEST") > -1, False)
        self.assertEqual(strings[2].find("TEST") > -1, True)
        self.assertEqual(strings[3].find("TEST") > -1, False)

if __name__ == '__main__':
    unittest.main()



