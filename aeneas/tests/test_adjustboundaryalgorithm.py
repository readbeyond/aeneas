#!/usr/bin/env python
# coding=utf-8

import unittest

from . import get_abs_path

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
        if a == None or b == None:
            return a == b
        if len(a) != len(b):
            return False
        for i in range(len(a)):
            rep_a = "%.3f %.3f" % (a[i][0], a[i][1])
            rep_b = "%.3f %.3f" % (b[i][0], b[i][1])
            if rep_a != rep_b:
                return False
        return True

    def run_aba(self, algorithm, value):
        aba = AdjustBoundaryAlgorithm(
            algorithm=algorithm,
            text_map=self.TEXT_MAP,
            speech=self.SPEECH,
            nonspeech=self.NONSPEECH,
            value=value
        )
        adjusted_map = aba.adjust()
        return self.maps_are_equal(adjusted_map, self.TEXT_MAP)

    def test_aba_auto_01(self):
        self.assertEqual(self.run_aba(AdjustBoundaryAlgorithm.AUTO, None), True)

    def test_aba_auto_02(self):
        self.assertEqual(self.run_aba(AdjustBoundaryAlgorithm.AUTO, "foo"), True)

    def test_aba_percent_01(self):
        self.assertEqual(self.run_aba(AdjustBoundaryAlgorithm.PERCENT, "0"), False)

    def test_aba_percent_02(self):
        self.assertEqual(self.run_aba(AdjustBoundaryAlgorithm.PERCENT, "25"), False)

    def test_aba_percent_03(self):
        self.assertEqual(self.run_aba(AdjustBoundaryAlgorithm.PERCENT, "50"), False)

    def test_aba_percent_04(self):
        self.assertEqual(self.run_aba(AdjustBoundaryAlgorithm.PERCENT, "75"), False)

    def test_aba_percent_05(self):
        self.assertEqual(self.run_aba(AdjustBoundaryAlgorithm.PERCENT, "100"), False)

    def test_aba_percent_06(self):
        # saturates at 0
        self.assertEqual(self.run_aba(AdjustBoundaryAlgorithm.PERCENT, "-50"), False)

    def test_aba_percent_07(self):
        # saturate at 100
        self.assertEqual(self.run_aba(AdjustBoundaryAlgorithm.PERCENT, "150"), False)
    
    def test_aba_percent_08(self):
        # defaults to 50
        self.assertEqual(self.run_aba(AdjustBoundaryAlgorithm.PERCENT, "foo"), False)

    def test_aba_rate_01(self):
        self.assertEqual(self.run_aba(AdjustBoundaryAlgorithm.RATE, "15"), False)

    def test_aba_rate_02(self):
        self.assertEqual(self.run_aba(AdjustBoundaryAlgorithm.RATE, "16"), False)

    def test_aba_rate_03(self):
        self.assertEqual(self.run_aba(AdjustBoundaryAlgorithm.RATE, "17"), False)

    def test_aba_rate_04(self):
        self.assertEqual(self.run_aba(AdjustBoundaryAlgorithm.RATE, "18"), False)

    def test_aba_rate_05(self):
        self.assertEqual(self.run_aba(AdjustBoundaryAlgorithm.RATE, "19"), False)

    def test_aba_rate_06(self):
        self.assertEqual(self.run_aba(AdjustBoundaryAlgorithm.RATE, "20"), False)

    def test_aba_rate_07(self):
        self.assertEqual(self.run_aba(AdjustBoundaryAlgorithm.RATE, "21"), False)

    def test_aba_rate_08(self):
        self.assertEqual(self.run_aba(AdjustBoundaryAlgorithm.RATE, "22"), False)

    def test_aba_rate_09(self):
        self.assertEqual(self.run_aba(AdjustBoundaryAlgorithm.RATE, "23"), True)

    def test_aba_rate_10(self):
        self.assertEqual(self.run_aba(AdjustBoundaryAlgorithm.RATE, "24"), True)

    def test_aba_rate_11(self):
        self.assertEqual(self.run_aba(AdjustBoundaryAlgorithm.RATE, "25"), True)

    def test_aba_rate_12(self):
        # defaults to 21
        self.assertEqual(self.run_aba(AdjustBoundaryAlgorithm.RATE, "foo"), False)

    def test_aba_aftercurrent_01(self):
        self.assertEqual(self.run_aba(AdjustBoundaryAlgorithm.AFTERCURRENT, "0.000"), False)

    def test_aba_aftercurrent_02(self):
        self.assertEqual(self.run_aba(AdjustBoundaryAlgorithm.AFTERCURRENT, "0.100"), False)

    def test_aba_aftercurrent_03(self):
        self.assertEqual(self.run_aba(AdjustBoundaryAlgorithm.AFTERCURRENT, "0.200"), False)

    def test_aba_aftercurrent_04(self):
        self.assertEqual(self.run_aba(AdjustBoundaryAlgorithm.AFTERCURRENT, "0.500"), False)

    def test_aba_aftercurrent_05(self):
        self.assertEqual(self.run_aba(AdjustBoundaryAlgorithm.AFTERCURRENT, "1.000"), False)

    def test_aba_aftercurrent_06(self):
        self.assertEqual(self.run_aba(AdjustBoundaryAlgorithm.AFTERCURRENT, "2.000"), False)

    def test_aba_aftercurrent_07(self):
        # defaults to current boundary
        self.assertEqual(self.run_aba(AdjustBoundaryAlgorithm.AFTERCURRENT, "foo"), True)

    def test_aba_beforenext_01(self):
        self.assertEqual(self.run_aba(AdjustBoundaryAlgorithm.BEFORENEXT, "0.000"), False)

    def test_aba_beforenext_02(self):
        self.assertEqual(self.run_aba(AdjustBoundaryAlgorithm.BEFORENEXT, "0.100"), False)

    def test_aba_beforenext_03(self):
        self.assertEqual(self.run_aba(AdjustBoundaryAlgorithm.BEFORENEXT, "0.200"), False)

    def test_aba_beforenext_04(self):
        self.assertEqual(self.run_aba(AdjustBoundaryAlgorithm.BEFORENEXT, "0.500"), False)

    def test_aba_beforenext_05(self):
        self.assertEqual(self.run_aba(AdjustBoundaryAlgorithm.BEFORENEXT, "1.000"), False)

    def test_aba_beforenext_06(self):
        self.assertEqual(self.run_aba(AdjustBoundaryAlgorithm.BEFORENEXT, "2.000"), False)

    def test_aba_beforenext_07(self):
        # defaults to current boundary
        self.assertEqual(self.run_aba(AdjustBoundaryAlgorithm.BEFORENEXT, "foo"), True)


if __name__ == '__main__':
    unittest.main()



