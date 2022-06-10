import lightgbm as lgbm
import pandas as pd
import pytest

from feature_draft.api import FeatureDraft
from feature_draft.estimator import LightGBMEstimator


@pytest.fixture
def lgbm_classification_model():
    return lgbm.LGBMClassifier()


@pytest.fixture
def classification_data():
    return pd.DataFrame({
        "a": [1, 2, 3, 4],
        "b": [1, 0, 1, 0],
    })


class TestFeatureDraft:

    def test_instantiation_with_light_gbm(
        self,
        lgbm_classification_model,
        classification_data,
    ):

        feature_draft = FeatureDraft(
            model=lgbm_classification_model,
            data=classification_data,
            features=["a"],
            response="b",
        )

        assert isinstance(
            feature_draft.estimator,
            LightGBMEstimator,
        )

    def test_get_metric_direction(
        self,
        lgbm_classification_model,
        classification_data,
    ):

        feature_draft = FeatureDraft(
            model=lgbm_classification_model,
            data=classification_data,
            features=["a"],
            response="b",
        )

        output = feature_draft._get_metric_direction()

        assert output == max

    def test_get_baseline_metric(
        self,
        lgbm_classification_model,
        classification_data,
    ):

        feature_draft = FeatureDraft(
            model=lgbm_classification_model,
            data=classification_data,
            features=["a"],
            response="b",
            cross_val_splits=5,
        )

        output = feature_draft._get_baseline_metric()

        assert output == [0.5] * 5
