import pytest
import numpy as np
from typing import Dict, List
from src.lm_from_scratch import LinearRegression


@pytest.fixture
def linear_regression_class():
    x_train_ = np.array([1, 2, 3, 4, 5]).reshape(5, 1)
    y_train_ = np.array([1, 4, 7, 9, 11])
    lm = LinearRegression(x_train_, y_train_)
    return lm


class TestLinearRegression:

    @pytest.fixture(autouse=True)
    def _linear_regression_class(self, linear_regression_class):
        self.lm = linear_regression_class

    def test_lm(self):
        assert isinstance(self.lm, LinearRegression)

    def test_r2(self):
        self.lm.train()
        assert self.lm.r_squared() == pytest.approx(0.923)

    def test_attibutes(self):
        assert isinstance(self.lm.intercept, int)
        assert isinstance(self.lm.old_cost, int)
        assert isinstance(self.lm.cutoff, float)
        assert isinstance(self.lm.lr, float)
        assert isinstance(self.lm.num_epochs, int)
        # assert isinstance(self.lm.callback_dict, Dict[str, List[int]])
        assert self.lm.r2 == 0
        assert isinstance(self.lm.r2, float)

    def test_run_epoch(self):
        assert self.lm.run_epoch()


