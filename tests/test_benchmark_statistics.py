import unittest
from tools.benchmark_statistics import paired_abba


def samples(round_values):
    return [{'round':i,'mode':mode,'gpu_us_per_operation':value}
            for i,values in enumerate(round_values)
            for mode,value in zip(['ggml','mojo','mojo','ggml'],values)]


class PairedStatisticsTests(unittest.TestCase):
    def test_known_constant_ratio(self):
        result=paired_abba(samples([[100,50,50,100]]*8),'mojo')
        self.assertAlmostEqual(result['geometric_mean_ratio'],2)

    def test_true_bias_is_not_removed(self):
        result=paired_abba(samples([[100,110,110,100]]*8),'mojo')
        self.assertAlmostEqual(result['geometric_mean_ratio'],100/110)

    def test_linear_drift_cancels_inside_abba(self):
        result=paired_abba(samples([[100+i,101+i,102+i,103+i] for i in range(8)]),'mojo')
        self.assertAlmostEqual(result['geometric_mean_ratio'],1)

    def test_abrupt_transition_remains_visible(self):
        values=[[100]*4]*4+[[100,120,120,120]]+[[120]*4]*3
        result=paired_abba(samples(values),'mojo')
        self.assertAlmostEqual(result['geometric_mean_ratio'],(110/120)**(1/8))
        self.assertLess(result['baseline_over_candidate_round_ratios'][4],.95)

    def test_incomplete_wrong_order_and_invalid_times_rejected(self):
        for change in ('missing','order','nan','zero'):
            data=samples([[100]*4]*8)
            if change=='missing':data.pop()
            elif change=='order':data[0]['mode']='mojo'
            elif change=='nan':data[0]['gpu_us_per_operation']=float('nan')
            else:data[0]['gpu_us_per_operation']=0
            with self.subTest(change=change),self.assertRaises(ValueError):paired_abba(data,'mojo')
