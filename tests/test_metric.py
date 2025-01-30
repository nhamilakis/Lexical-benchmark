from lexical_benchmark.stats.metric import Metric
import unittest
from lexical_benchmark.datasets import utils as dataset_utils



class TestMetricCalculation(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        # Sample text data
        self.text_data = [
            "the cat sat on the mat",
            "the dog ran in the park",
            "a cat and dog play together",
        ]
        
        # Sample CDI words and counts
        self.cdi_words = ['cat', 'dog', 'play', 'run']
        self.word_dict = dataset_utils.DictionairyCleaner(lang="EN")
        
        # Test parameters
        self.test_params = {
            'threshold': 1,
            'word_count_est': 1000,
            'chunk_size': 5,
            'metric_lst': ['type_token_ratio', 'rej_type_rate', 'CDI']
        }

    def test_ttr_calculation(self):
        """Test type-token ratio calculation"""
        metric = Metric(
            data=self.text_data,
            metric_lst=['type_token_ratio'],
            chunk_size=self.test_params['chunk_size']
        )
        ttr = metric.compute_ttr()
        
        # TTR should be between 0 and 1
        self.assertGreaterEqual(ttr, 0)
        self.assertLessEqual(ttr, 1)
        
        # For our sample data:
        # Unique words: cat, dog, play, run, the, on, mat, in, park, a, and, together
        # Total words: 17
        # Expected TTR should be around 12/17 ≈ 0.706
        self.assertAlmostEqual(ttr, 12/17, places=2)

    def test_rejection_rate(self):
        """Test rejection rate calculation"""
        metric = Metric(
            data=self.text_data,
            metric_lst=['rej_type_rate'],
            chunk_size=self.test_params['chunk_size']
        )
        rej_rate = metric.compute_type_rej_rate()
        
        # Rejection rate should be between 0 and 1
        self.assertGreaterEqual(rej_rate, 0)
        self.assertLessEqual(rej_rate, 1)

    def test_cdi_score(self):
        """Test CDI score calculation"""
        metric = Metric(
            data=self.text_data,
            metric_lst=['CDI'],
            CDI_words=self.cdi_words,
            threshold=self.test_params['threshold'],
            word_count_est=self.test_params['word_count_est'],
            word_count=list(self.word_dict.values())
        )
        cdi_score = metric.compute_CDI()
        
        # CDI score should be between 0 and 100
        self.assertGreaterEqual(cdi_score, 0)
        self.assertLessEqual(cdi_score, 100)
        
        # For our sample data:
        # cat (2 occurrences) -> 1
        # dog (2 occurrences) -> 1
        # play (1 occurrence) -> 1
        # run (0 occurrences) -> 0
        # Expected score: (1 + 1 + 1 + 0)/4 * 100 = 75
        self.assertEqual(cdi_score, 75)

    def test_compute_all_metrics(self):
        """Test computation of all metrics together"""
        metric = Metric(
            data=self.text_data,
            temp="test",
            metric_lst=self.test_params['metric_lst'],
            threshold=self.test_params['threshold'],
            CDI_words=self.cdi_words,
            word_count=list(self.word_dict.values()),
            word_count_est=self.test_params['word_count_est'],
            chunk_size=self.test_params['chunk_size']
        )
        
        results = metric.compute_metrics()
        
        # Check results structure
        self.assertEqual(len(results), 4)  # temp + 3 metrics
        self.assertEqual(results[0], "test")  # temperature
        
        # Check that all metrics are numbers
        for value in results[1:]:
            self.assertIsInstance(value, (int, float, type(None)))

if __name__ == '__main__':
    unittest.main(verbosity=2)