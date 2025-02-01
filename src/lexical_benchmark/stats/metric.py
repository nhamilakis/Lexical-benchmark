import functools
import typing as t

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from lexical_benchmark.datasets import childes
from lexical_benchmark.datasets import utils as dataset_utils
from lexical_benchmark.stats import normalised_rejection_rates
from lexical_benchmark.stats.CDI_scores import CDICalculator


class MetricAnalyzer:
    """Compute and manage metrics for both model-generated and human data."""

    def __init__(
        self,
        metric_lst: list[str],
        threshold: int,
        CDI_words: list[str],
        chunk_size: int,
        word_dict: dict[str, t.Any],
        word_count_est: int,
        CDI_month_dict: dict[str, dict],
    ) -> None:
        """Initialize MetricsComputer with configuration parameters."""
        self.metric_lst = metric_lst
        self.threshold = threshold
        self.CDI_words = CDI_words
        self.chunk_size = chunk_size
        self.word_dict = word_dict
        self.word_count_est = word_count_est
        self.CDI_month_dict = CDI_month_dict

        # Common metric indices mapping
        self.metric_indices = {"type_token_ratio": 1, "rej_type_rate": 2, "rej_token_rate": 3, "CDI": 4}

    def compute_metrics_base(
        self,
        sent_lst: list[str],
        temp: str | float,
        previous_words: dict[str, int],
    ) -> tuple[list, dict[str, dict]]:
        """Base function for computing metrics."""
        metric = Metric(
            data=sent_lst,
            temp=temp,
            metric_lst=self.metric_lst,
            threshold=self.threshold,
            CDI_words=self.CDI_words,
            word_count_est=self.word_count_est,
            chunk_size=self.chunk_size,
            word_dict=self.word_dict,
            previous_words=previous_words,
        )
        return metric.compute_metrics()

    def compute_model_metrics(
        self,
        gen: pd.DataFrame,
        temp_lst: list[float],
        info_dict: dict[str, str | int],
    ) -> tuple[pd.DataFrame, dict[str, dict]]:
        """Compute model-specific metrics."""
        scores = []

        for temp in temp_lst:
            word_dict_manager = WordDictManager(
                dataset=info_dict["dataset"],
                chunk=info_dict["chunk"],
                model_type=info_dict["model_type"],
                temp=temp,
            )

            previous_words = word_dict_manager.load_word_dict(self.CDI_month_dict)
            sent_lst = gen[f"unprompted_{temp}"].apply(char2word).tolist()

            metric_results, cum_counts = self.compute_metrics_base(
                sent_lst=sent_lst,
                temp=temp,
                previous_words=previous_words,
            )

            if metric_results:
                scores.append(self._create_metric_row(temp, metric_results))

            if cum_counts is not None:
                self.CDI_month_dict = word_dict_manager.write_word_dict(cum_counts, self.CDI_month_dict)

        if not scores:
            return pd.DataFrame(), self.CDI_month_dict

        score = pd.DataFrame(scores, columns=["temp"] + self.metric_lst)
        score = score.assign(**info_dict)
        score["word_num"] = gen["sent_len"].sum()

        return score, self.CDI_month_dict

    def compute_human_metrics(
        self,
        ref_data: pd.DataFrame,
        word_est_dict: dict[int, float],
        CDI: bool,
        previous_words: dict[str, int],
    ) -> pd.DataFrame:
        """Compute human metrics."""
        info_dict = {"dataset": "CHILDES", "chunk": "00", "model_type": "human", "temp": "1.0"}

        gen_grouped = ref_data.groupby("month")
        scores = []

        for month, gen in gen_grouped:
            word_dict_manager = WordDictManager(
                dataset=info_dict["dataset"],
                chunk=info_dict["chunk"],
                model_type=info_dict["model_type"],
                temp=info_dict["temp"],
            )

            word_count_est = load_word_count_est(word_est_dict, month, CDI)
            previous_words = word_dict_manager.load_word_dict(self.CDI_month_dict)
            sent_lst = gen["text"].tolist()

            metric_results, cum_counts = self.compute_metrics_base(
                sent_lst=sent_lst,
                temp=month,
                previous_words=previous_words,
            )

            if metric_results:
                row = self._create_metric_row(month, metric_results)
                row.append(gen["sent_len"].sum())
                scores.append(row)

            if cum_counts is not None:
                self.CDI_month_dict = word_dict_manager.write_word_dict(cum_counts, self.CDI_month_dict)

        if not scores:
            return pd.DataFrame()

        score = pd.DataFrame(scores, columns=["month"] + self.metric_lst + ["word_num"])
        score = score.assign(**info_dict)

        return score

    def _create_metric_row(self, identifier: float | str, metric_results: list) -> list:
        """Helper method to create a metric row."""
        row = [identifier]
        for metric_name in self.metric_lst:
            idx = self.metric_indices.get(metric_name)
            if idx is not None and idx < len(metric_results):
                row.append(metric_results[idx])
            else:
                row.append(None)
        return row


class Metric:
    def __init__(
        self,
        data: list[str],
        temp: str = None,
        metric_lst: list = None,
        threshold: int = None,  # count_based threshold for CDI score
        CDI_words: list = None,  # a list of selected CDI words
        word_count_est: int = None,  # count estimation based on prior study
        chunk_size: int | None = None,
        word_dict: dataset_utils.DictionairyCleaner | None = None,
        previous_words: dict[str, int] = None,
    ) -> None:
        self.temp = temp
        self.metric_lst = metric_lst or []
        self.threshold = threshold
        self.CDI_words = CDI_words or []
        self.chunk_size = chunk_size
        self.word_count_est = word_count_est
        self.word_dict = word_dict
        self.previous_words = previous_words if previous_words is not None else {}
        self.data = segment_sent(data)

        if word_dict is None:
            word_dict = dataset_utils.DictionairyCleaner(lang="EN")
        self.word_clean_fn = functools.partial(word_clean_fn, word_dict=word_dict)

        if chunk_size:
            chunks = normalised_rejection_rates.chunk_splitter(self.data, chunk_size)
            self.stats = normalised_rejection_rates.clean_chunk_list(chunks=chunks, filter_fn=self.word_clean_fn)
            self.chunked = True
        else:
            self.chunked = False
            self.stats = normalised_rejection_rates.word_clean_chunk(self.data, filter_fn=self.word_clean_fn)

    def compute_ttr(self) -> float:
        if self.chunked:
            return self.stats.mean_type_token_ratio()
        return self.stats.type_token_ratio()

    def compute_type_rej_rate(self) -> float:
        if self.chunked:
            return self.stats.mean_type_rejection_rate()
        return self.stats.type_rejection_rate()

    def compute_token_rej_rate(self) -> float:
        if self.chunked:
            return self.stats.mean_token_rejection_rate()
        return self.stats.token_rejection_rate()

    def compute_CDI(self) -> float:
        # Create calculator instance
        calculator = CDICalculator(
            CDI_words=self.CDI_words,
            threshold=self.threshold,
            word_count_est=self.word_count_est,
            word_list=self.data,
            previous_words=self.previous_words,
        )
        # Calculate mean score
        cum_counts, mean_score = calculator.compute_mean_cdi_score()

        return mean_score, cum_counts

    def compute_metrics(self) -> list:
        row = [self.temp]
        row.extend(
            [
                self.compute_ttr() if "type_token_ratio" in self.metric_lst else None,
                self.compute_type_rej_rate() if "rej_type_rate" in self.metric_lst else None,
                self.compute_token_rej_rate() if "rej_token_rate" in self.metric_lst else None,
                self.compute_CDI()[0] if "CDI" in self.metric_lst else None,
            ]
        )
        cum_counts = self.compute_CDI()[1] if "CDI" in self.metric_lst else None
        return row, cum_counts


class WordDictManager:
    """Manage word dictionary operations with nested structure."""

    def __init__(self, dataset: str, chunk: str | int, model_type: str, temp: str | float) -> None:
        """Initialize WordDictManager with all values converted to strings."""
        self.dataset = str(dataset)
        self.chunk = str(chunk)
        self.model_type = str(model_type)
        self.temp = str(temp)

    def load_word_dict(self, CDI_month_dict: dict[str, dict]) -> dict[str, int]:
        """Load dictionary based on initialized parameters."""
        try:
            return CDI_month_dict.get(self.dataset, {}).get(self.chunk, {}).get(self.model_type, {}).get(self.temp, {})
        except Exception as e:
            print(f"Error accessing dictionary: {e}")
            print(f"Path: {self.dataset}/{self.chunk}/{self.model_type}/{self.temp}")
            return {}

    def write_word_dict(self, previous_words: dict[str, dict], CDI_month_dict: dict[str, dict]) -> dict[str, dict]:
        """Write dictionary based on initialized parameters."""
        # Initialize empty dictionaries if they don't exist
        if not isinstance(CDI_month_dict, dict):
            CDI_month_dict = {}

        if self.dataset not in CDI_month_dict:
            CDI_month_dict[self.dataset] = {}

        if self.chunk not in CDI_month_dict[self.dataset]:
            CDI_month_dict[self.dataset][self.chunk] = {}

        if self.model_type not in CDI_month_dict[self.dataset][self.chunk]:
            CDI_month_dict[self.dataset][self.chunk][self.model_type] = {}

        # Store the words
        CDI_month_dict[self.dataset][self.chunk][self.model_type][self.temp] = previous_words

        return CDI_month_dict


class SigmoidFitter:
    def __init__(self, x_data: list[int], y_data: list[int], target_y: float) -> None:
        self.x_data = x_data
        self.y_data = y_data
        self.target_y = target_y

    def sigmoid(self, x, a, b):
        return 1 / (1 + np.exp(-(a * x + b)))

    def fit_sigmoid(self):
        popt, _ = curve_fit(self.sigmoid, self.x_data, self.y_data, maxfev=100000, method="trf")
        x_fit = np.linspace(0, max(self.x_data), 40)
        y_fit = self.sigmoid(x_fit, *popt)

        if max(self.y_data) < self.target_y:
            while y_fit[-1] < self.target_y:
                x_fit = np.append(x_fit, x_fit[-1] + 1)
                y_fit = np.append(y_fit, self.sigmoid(x_fit[-1], *popt))
                if y_fit[-1] >= self.target_y:
                    break

        target_y_index = np.argmin(np.abs(y_fit - self.target_y))
        target_x = x_fit[target_y_index]

        return {"target_x": target_x, "slope": popt[0], "offset": popt[1]}


def load_dict(dataset_name: str):
    """Load dictionary based on different datasets."""
    if dataset_name == "child":
        print("Append en_dict with adult input")
        dataset = childes.CHILDESDataset()
        childes_adult_extras_lexique = childes.CHILDESExtrasLexicon(dataset)
        childes_adult_extras_lexique.add_lang("Eng-NA", "adult")
        childes_adult_extras_lexique.add_lang("Eng-UK", "adult")
        dict_hash_id = childes_adult_extras_lexique.cache_current()
        en_dict = dataset_utils.DictionairyCleaner(lang="EN", childes_extra_id=dict_hash_id)
    else:
        en_dict = dataset_utils.DictionairyCleaner(lang="EN")
    print("Dictionary has been loaded!")
    return en_dict


def word_clean_fn(word: str, word_dict: dataset_utils.DictionairyCleaner) -> bool:
    """Check if a word is in dict."""
    return word_dict.check(word)


def segment_sent(data: list[str]) -> list[str]:
    """Segment sentence list into word list if needed."""
    if isinstance(data[0], (str, bytes)):
        return [word for sent in data for word in str(sent).split()]
    return data
