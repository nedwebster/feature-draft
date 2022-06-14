from numbers import Number
import operator
from typing import List

import pandas as pd
import numpy as np
from scipy import stats
from sigfig import round

from feature_draft import estimator, cross_val
from feature_draft.configs.metrics_config import METRICS_CONFIG


class FeatureDraft:

    def __init__(
        self,
        model,
        data: pd.DataFrame,
        features: List[str],
        response: str,
        cross_val_splits: int = 5,
    ):
        self.estimator = estimator.build_estimator(model)
        self.cross_validator = cross_val.CrossValidator(
            n_splits=cross_val_splits,
            stratified=self.estimator.is_classification(),
        )
        self.data = data
        self.response = response
        self.candidate_features = features.copy()
        self.selected_features = []
        self.best_metrics = self._get_baseline_metric()
        self.metric_direction = self._get_metric_direction()

    # TODO: Move metric code to seprate metrics class
    def _get_metric_direction(self):
        """
        Determine whether the estimator evaluation metric
        is increasing or decreasing, and assign the appropriate
        function.

        """
        return METRICS_CONFIG[self.estimator.metric]

    def _get_baseline_metric(self):
        """
        Generate a baseline metric to compare against the first round of
        feature selection. Returns a k length array, where k is the number
        of splits used in the cross-validator.

        """
        avg_response = self.data[self.response].mean()
        avg_response_array = [avg_response] * self.data.shape[0]

        baseline_metric = self.estimator.evaluate(
            self.data[self.response],
            avg_response_array,
        )

        return [baseline_metric] * self.cross_validator.n_splits

    def _build_with_candidate_feature(self, feature: str):
        """
        Generate cross-validation metrics with the current selected features
        and a new candidate feature.

        """

        candidate_features = self.selected_features.copy() + [feature]

        return self.cross_validator.cross_validation_build(
            X=self.data[candidate_features],
            y=self.data[self.response],
            model=self.estimator,
        )

    def _get_best_feature(self, feature_results: dict):
        """Get the best feature from a dictionary of feature metrics"""

        feature_means = {k: np.mean(v) for k, v in feature_results.items()}

        return self.metric_direction(
            feature_means.items(),
            key=operator.itemgetter(1)
        )[0]

    # TODO: probably should move these checks outside of the FeatureDraft class
    def _check_feature_versus_current(self, feature_results):
        """
        Evaluate metric improvement for the best feature. This compares
        the metric results against the current saved best metrics.

        """

        # test average metric movement is an improvement
        if self.metric_direction(
            np.mean(self.best_metrics),
            np.mean(feature_results),
        ) != np.mean(feature_results):
            return False

        # test difference is statistically significant
        test_results = stats.ttest_ind(feature_results, self.best_metrics)
        if test_results.pvalue < 0.05:
            return True
        else:
            return False

    def draft_round(self):
        """
        Perform a round of the feature selection draft, choosing
        the next best feature to add to the model.

        """

        feature_results = {}

        # loop over all candidate features
        for feature in self.candidate_features:

            # perform cv build with additional feature
            feature_results[feature] = self._build_with_candidate_feature(
                feature
            )

        # identify the best feature from all candidate features
        best_feature = self._get_best_feature(feature_results)

        # check best_feature improves the model
        is_improvement = self._check_feature_versus_current(
            feature_results[best_feature]
        )

        if is_improvement:
            return (best_feature, feature_results[best_feature])

        else:
            return None

    def _update_feature_lists(self, feature, feature_result):
        """
        Remove the selected feature from the list of candidates,
        add it to the final selected list, and update the best metrics.
        """

        self.selected_features.append(feature)
        self.candidate_features.remove(feature)
        self.best_metrics = feature_result

        if len(self.candidate_features) == 0:
            self.metric_improving = False

    def _calcualte_metric_improvement(self, feature_result: List[Number]):

        metric_improvement = (
            np.mean(feature_result) - np.mean(self.best_metrics)
        )

        return metric_improvement

    def draft_features(self):

        print(f"Baseline Metric: {np.mean(self.best_metrics)}")

        self.metric_improving = True
        i = 1

        while self.metric_improving:

            print(f"\nDraft Round: {i}")

            # run feature draft round
            output = self.draft_round()

            # end loop if no feature is selected
            if output is None:
                self.metric_improving = False

            # update feature lists if feature is selected
            else:
                metric_improvement = self._calcualte_metric_improvement(
                    output[1]
                )
                self._update_feature_lists(
                    output[0],
                    output[1],
                )

                # TODO: Move verbose printing from here
                print(
                    f"Feature Selected: {output[0]},"
                    f"\nMetric Improvement: {round(metric_improvement, 5)}"
                    f"\nNew Metric: {round(np.mean(output[1]), 5)}"
                )
                i += 1

        print(f"Draft finished, final feature list: {self.selected_features}")
