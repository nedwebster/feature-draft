from typing import List

import pandas as pd
import numpy as np
from sigfig import round

from feature_draft import estimator, cross_val, feature_scouter


class FeatureDraft:

    def __init__(
        self,
        model,
        data: pd.DataFrame,
        features: List[str],
        response: str,
        cross_val_splits: int = 5,
        alpha=0.05,
    ):
        self.estimator = estimator.build_estimator(model)
        self.cross_validator = cross_val.CrossValidator(
            n_splits=cross_val_splits,
            stratified=self.estimator.is_classification(),
        )

        self.scout = feature_scouter.Scout(
            metric=self.estimator.metric,
            alpha=alpha,
        )

        self.data = data
        self.response = response
        self.candidate_features = features.copy()
        self.selected_features = []
        self.best_metrics = self._get_baseline_metric()

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

    def _update_feature_lists(self, feature, feature_result):
        """
        Remove the selected feature from the list of candidates,
        add it to the final selected list, and update the best metrics.
        """

        self.selected_features.append(feature)
        self.candidate_features.remove(feature)
        self.best_metrics = feature_result

        # stop process if there are no more candidate features
        if len(self.candidate_features) == 0:
            self.metric_improving = False

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

        output = self.scout.evaluate_results(
            feature_results=feature_results,
            current_results=self.best_metrics,
        )

        return output

    def draft_features(self):

        print(f"Baseline Metric: {np.mean(self.best_metrics)}")

        self.metric_improving = True
        i = 1

        while self.metric_improving:

            print(f"\nDraft Round: {i}")

            # run feature draft round
            draft_results = self.draft_round()

            # end loop if no feature is selected
            if draft_results is None:
                self.metric_improving = False

            # update feature lists if feature is selected
            else:
                metric_improvement = self.scout._get_metric_improvement(
                    draft_results[1],
                    self.best_metrics,
                )
                self._update_feature_lists(
                    draft_results[0],
                    draft_results[1],
                )

                # TODO: Move verbose printing from here
                print(
                    f"Feature Selected: {draft_results[0]},"
                    f"\nMetric Improvement: {round(metric_improvement, 5)}"
                    f"\nNew Metric: {round(np.mean(draft_results[1]), 5)}"
                )
                i += 1

        print(f"Draft finished, final feature list: {self.selected_features}")
