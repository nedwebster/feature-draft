import lightgbm as lgbm
import pandas as pd
import pytest
from sklearn.model_selection import KFold, StratifiedKFold

from feature_draft import cross_val, estimator


class TestCrossValidator:

    def test_instantiation(self):

        _ = cross_val.CrossValidator()

    @pytest.mark.parametrize(
        "stratified, kfold_type",
        [
            (True, StratifiedKFold),
            (False, KFold),
        ]
    )
    def test_set_fold_object(self, stratified, kfold_type):

        test_cv = cross_val.CrossValidator(n_splits=3, stratified=stratified)

        assert isinstance(test_cv.fold_object, kfold_type)
        assert test_cv.fold_object.n_splits == 3

    def test_cross_validation_build(self, mocker):

        data = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [2, 3, 4, 5, 6],
            "c": [0, 0, 1, 1, 1],
        })

        model = estimator.LightGBMEstimator(lgbm.LGBMClassifier())

        mocker.patch.object(
            estimator.LightGBMEstimator,
            "fit_and_evaluate",
            return_value=1,
        )

        test_cv = cross_val.CrossValidator(n_splits=3)

        output = test_cv.cross_validation_build(
            X=data[["a", "b"]],
            y=data["c"],
            model=model,
        )

        assert output == [1, 1, 1]
