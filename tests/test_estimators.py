import lightgbm as lgbm
import pytest
from sklearn import metrics

from feature_draft import estimator


class TestBaseEstimator():
    """Tests for the feature_draft.estimator.BaseEstimator class"""

    def test_estimator_instantiation(self):

        _ = estimator.BaseEstimator(estimator=None)

    def test_estimator_setter_error(self):

        with pytest.raises(TypeError):
            _ = estimator.BaseEstimator(estimator="test")

    def test_metric_setter_error(self):

        with pytest.raises(ValueError):
            _ = estimator.BaseEstimator(estimator=None, metric="test")


class TestLightGBMEstimator():
    """Tests for the feature_draft.estimator.LightGBMEstimator class"""

    def test_estimator_instantiation(self):

        _ = estimator.LightGBMEstimator(estimator=lgbm.LGBMClassifier())

    def test_estimator_metric(self):

        test_estimator = estimator.LightGBMEstimator(
            estimator=lgbm.LGBMClassifier()
        )

        assert test_estimator.metric == metrics.roc_auc_score
