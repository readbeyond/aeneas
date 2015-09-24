#!/usr/bin/env python
# coding=utf-8

import unittest

from aeneas.adjustboundaryalgorithm import AdjustBoundaryAlgorithm

class TestAdjustBoundaryAlgorithm(unittest.TestCase):

    TEXT_MAP = [
        [0.000, 2.720, u"f000001", u"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"],
        [2.720, 7.000, u"f000002", u"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"],
        [7.000, 8.880, u"f000003", u"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"],
        [8.880, 10.160, u"f000004", u"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"],
        [10.160, 13.000, u"f000005", u"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"],
        [13.000, 16.480, u"f000006", u"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"],
        [16.480, 19.760, u"f000007", u"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"],
        [19.760, 22.640, u"f000008", u"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"],
        [22.640, 24.480, u"f000009", u"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"],
        [24.480, 26.920, u"f000010", u"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"],
        [26.920, 30.840, u"f000011", u"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"],
        [30.840, 36.560, u"f000012", u"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"],
        [36.560, 38.640, u"f000013", u"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"],
        [38.640, 40.960, u"f000014", u"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"],
        [40.960, 44.880, u"f000015", u"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"],
        [44.880, 49.400, u"f000016", u"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"],
        [49.400, 53.840, u"f000017", u"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"],
        [53.840, 56.120, u"f000018", u"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"]
    ]

    SPEECH = [
        [0.120, 2.240],
        [2.440, 2.560],
        [2.880, 4.760],
        [5.520, 6.840],
        [7.040, 9.680],
        [9.880, 9.920],
        [10.120, 10.240],
        [10.440, 11.760],
        [12.040, 12.880],
        [13.920, 15.520],
        [15.720, 15.840],
        [16.480, 17.960],
        [18.200, 18.720],
        [19.800, 21.040],
        [21.440, 22.280],
        [22.640, 24.600],
        [25.080, 26.560],
        [26.800, 26.840],
        [27.840, 28.520],
        [28.720, 29.560],
        [29.920, 30.680],
        [30.880, 34.400],
        [34.800, 35.880],
        [36.080, 36.200],
        [36.560, 38.360],
        [39.240, 40.400],
        [40.960, 42.840],
        [43.040, 43.600],
        [43.880, 44.200],
        [44.440, 44.720],
        [45.440, 46.320],
        [46.960, 47.920],
        [48.240, 48.400],
        [48.680, 49.040],
        [49.400, 50.040],
        [51.040, 51.840],
        [52.400, 52.600],
        [52.800, 53.480],
        [53.720, 54.520],
        [54.920, 55.400]
    ]
    NONSPEECH = [
        [0.000, 0.120],
        [2.240, 2.440],
        [2.560, 2.880],
        [4.760, 5.520],
        [6.840, 7.040],
        [9.680, 9.880],
        [9.920, 10.120],
        [10.240, 10.440],
        [11.760, 12.040],
        [12.880, 13.920],
        [15.520, 15.720],
        [15.840, 16.480],
        [17.960, 18.200],
        [18.720, 19.800],
        [21.040, 21.440],
        [22.280, 22.640],
        [24.600, 25.080],
        [26.560, 26.800],
        [26.840, 27.840],
        [28.520, 28.720],
        [29.560, 29.920],
        [30.680, 30.880],
        [34.400, 34.800],
        [35.880, 36.080],
        [36.200, 36.560],
        [38.360, 39.240],
        [40.400, 40.960],
        [42.840, 43.040],
        [43.600, 43.880],
        [44.200, 44.440],
        [44.720, 45.440],
        [46.320, 46.960],
        [47.920, 48.240],
        [48.400, 48.680],
        [49.040, 49.400],
        [50.040, 51.040],
        [51.840, 52.400],
        [52.600, 52.800],
        [53.480, 53.720],
        [54.520, 54.920],
        [55.400, 56.160]
    ]

    def maps_are_equal(self, a, b):
        if (a is None) or (b is None):
            return a == b
        if len(a) != len(b):
            return False
        for i in range(len(a)):
            rep_a = "%.3f %.3f" % (a[i][0], a[i][1])
            rep_b = "%.3f %.3f" % (b[i][0], b[i][1])
            if rep_a != rep_b:
                return False
        return True

    def run_aba(self, algorithm, value, expected):
        aba = AdjustBoundaryAlgorithm(
            algorithm=algorithm,
            text_map=self.TEXT_MAP,
            speech=self.SPEECH,
            nonspeech=self.NONSPEECH,
            value=value
        )
        adjusted_map = aba.adjust()
        self.assertEqual(self.maps_are_equal(adjusted_map, self.TEXT_MAP), expected)

    def test_auto(self):
        tests = [
            [None, True],
            ["foo", True]
        ]
        for test in tests:
            self.run_aba(AdjustBoundaryAlgorithm.AUTO, test[0], test[1])

    def test_percent(self):
        tests = [
            ["0", False],
            ["25", False],
            ["50", False],
            ["75", False],
            ["100", False],
            ["-50", False], # saturates at 0
            ["150", False], # saturates at 100
            ["foo", False]  # defaults to 50
        ]
        for test in tests:
            self.run_aba(AdjustBoundaryAlgorithm.PERCENT, test[0], test[1])

    def test_rate(self):
        tests = [
            ["15", False],
            ["16", False],
            ["17", False],
            ["18", False],
            ["19", False],
            ["20", False],
            ["21", False],
            ["22", False],
            ["23", True],
            ["24", True],
            ["25", True],
            ["-50", False], # defaults to 21
            ["0", False],   # defaults to 21
            ["foo", False]  # defaults to 21
        ]
        for test in tests:
            self.run_aba(AdjustBoundaryAlgorithm.RATE, test[0], test[1])

    def test_rateaggressive(self):
        tests = [
            ["15", False],
            ["16", False],
            ["17", False],
            ["18", False],
            ["19", False],
            ["20", False],
            ["21", False],
            ["22", False],
            ["23", False],
            ["24", True],
            ["25", True],
            ["-50", False], # defaults to 21
            ["0", False],   # defaults to 21
            ["foo", False]  # defaults to 21
        ]
        for test in tests:
            self.run_aba(AdjustBoundaryAlgorithm.RATEAGGRESSIVE, test[0], test[1])

    def test_aftercurrent(self):
        tests = [
            ["0.000", True],
            ["0.100", False],
            ["0.200", False],
            ["0.500", False],
            ["1.000", False],
            ["2.000", False],
            ["-1", True], # defaults to current boundary
            ["foo", True] # defaults to current boundary
        ]
        for test in tests:
            self.run_aba(AdjustBoundaryAlgorithm.AFTERCURRENT, test[0], test[1])

    def test_beforenext(self):
        tests = [
            ["0.000", True],
            ["0.100", False],
            ["0.200", False],
            ["0.500", False],
            ["1.000", False],
            ["2.000", False],
            ["-1", True], # defaults to current boundary
            ["foo", True] # defaults to current boundary
        ]
        for test in tests:
            self.run_aba(AdjustBoundaryAlgorithm.BEFORENEXT, test[0], test[1])

    def test_offset(self):
        tests = [
            ["-100.000", False],
            ["-2.000", False],
            ["-1.000", False],
            ["-0.500", False],
            ["-0.200", False],
            ["-0.100", False],
            ["0.000", True],
            ["0.100", False],
            ["0.200", False],
            ["0.500", False],
            ["1.000", False],
            ["2.000", False],
            ["100.000", False],
            ["foo", True] # defaults to current boundary
        ]
        for test in tests:
            self.run_aba(AdjustBoundaryAlgorithm.OFFSET, test[0], test[1])

if __name__ == '__main__':
    unittest.main()



