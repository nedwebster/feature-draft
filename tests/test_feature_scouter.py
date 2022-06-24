import pytest
from sklearn import metrics

from feature_draft import feature_scouter


class TestScout():

    def test_instantiation(self):

        output = feature_scouter.Scout(metric=metrics.roc_auc_score)

        assert isinstance(output, feature_scouter.Scout)
        assert hasattr(output, "metric_direction")

    @pytest.mark.parametrize(
        'metric, direction',
        [
            (metrics.roc_auc_score, max),
            (metrics.mean_squared_error, min),
        ],
    )
    def test_get_metric_direction(self, metric, direction):

        test_scout = feature_scouter.Scout(metric=metric)

        output = test_scout._get_metric_direction()

        assert output == direction

    def test_get_best_features(self):

        test_scout = feature_scouter.Scout(metric=metrics.roc_auc_score)

        feature_results = {
            "feature_1": [0.2, 0.3],
            "feature_2": [0.3, 0.4],
            "feature_3": [0.4, 0.5],
        }

        output = test_scout._get_best_feature(feature_results)

        assert output[0] == "feature_3"
        assert output[1] == [0.4, 0.5]

    @pytest.mark.parametrize(
        'a, b, result',
        [
            ([1, 1, 1], [0, 0, 0], True),
            ([1, 1, 1], [1, 1, 1], False),
        ],
    )
    def test_compute_ttest(self, a, b, result):
        test_scout = feature_scouter.Scout(metric=metrics.roc_auc_score)

        output = test_scout._compute_ttest(a, b)

        assert output is result

    @pytest.mark.parametrize(
        'a, b, result',
        [
            ([0, 0, 0], [1, 1, 1], True),
            ([2, 2, 2], [1, 1, 1], False),
        ],
    )
    def test_check_feature_versus_current(self, a, b, result, mocker):

        mocker.patch.object(
            feature_scouter.Scout,
            "_compute_ttest",
            return_value=True,
        )

        test_scout = feature_scouter.Scout(metric=metrics.roc_auc_score)

        output = test_scout._check_feature_versus_current(b, a)

        assert output is result

    @pytest.mark.parametrize(
        'a, b, result',
        [
            ([0, 0, 0], [1, 1, 1], 1),
            ([3, 3, 3], [1, 1, 1], 2),
        ],
    )
    def test_get_metric_improvement(self, a, b, result):

        test_scout = feature_scouter.Scout(metric=metrics.roc_auc_score)

        output = test_scout._get_metric_improvement(a, b)

        assert output == result

    def test_evaluate_results_is_improvement(self):

        feature_results = {
            "feature_1": [1, 1, 1],
            "feature_2": [2, 2, 2],
        }

        current_results = [0, 0, 0]

        test_scout = feature_scouter.Scout(metric=metrics.roc_auc_score)

        output = test_scout.evaluate_results(feature_results, current_results)

        assert output[0] == "feature_2"
        assert output[1] == [2, 2, 2]

    def test_evaluate_results_no_improvement(self):

        feature_results = {
            "feature_1": [1, 1, 1],
            "feature_2": [2, 2, 2],
        }

        current_results = [3, 3, 3]

        test_scout = feature_scouter.Scout(metric=metrics.roc_auc_score)

        output = test_scout.evaluate_results(feature_results, current_results)

        assert output is None
