import operator

import numpy as np
from scipy import stats

from feature_draft.configs.metrics_config import METRICS_CONFIG


class Scout:

    def __init__(self, metric, alpha=0.05):

        self.metric = metric
        self.metric_direction = self._get_metric_direction()
        self.alpha = alpha

    def _get_metric_direction(self):
        """
        Determine whether the estimator evaluation metric
        is increasing or decreasing, and assign the appropriate
        function.

        """
        return METRICS_CONFIG[self.metric]

    def _get_best_feature(self, feature_results: dict):
        """Get the best feature from a dictionary of feature metrics"""

        feature_means = {k: np.mean(v) for k, v in feature_results.items()}

        best_feature = self.metric_direction(
            feature_means.items(),
            key=operator.itemgetter(1)
        )[0]

        return best_feature, feature_results[best_feature]

    def _compute_ttest(self, best_feature_results, current_results):
        """Compute ttest to assess if feature is statistically significant"""

        test_results = stats.ttest_ind(
            best_feature_results,
            current_results,
        )

        if test_results.pvalue < self.alpha:
            return True
        else:
            return False

    def _check_feature_versus_current(
        self,
        best_feature_results,
        current_results,
    ) -> bool:
        """
        Evaluate metric improvement for the best feature. This compares
        the metric results against the current best metrics.

        """

        # test average metric movement is an improvement
        if self.metric_direction(
            np.mean(current_results),
            np.mean(best_feature_results),
        ) != np.mean(best_feature_results):
            return False

        return self._compute_ttest(best_feature_results, current_results)

    def _get_metric_improvement(
        self,
        best_feature_results,
        current_results,
    ):
        """Calculate difference in metrics from two sets of results"""

        metric_improvement = abs(
            np.mean(best_feature_results) - np.mean(current_results)
        )

        return metric_improvement

    def evaluate_results(self, feature_results, current_results):
        """
        Find the best feature from a dict of feature results, then assess
        whether it is an improvement over the current results.

        """

        best_feature, best_feature_results = self._get_best_feature(
            feature_results
        )

        if self._check_feature_versus_current(
            best_feature_results,
            current_results,
        ):
            return best_feature, best_feature_results

        else:
            return None
