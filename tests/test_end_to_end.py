from datetime import date

from liq.examples.data.fixtures import btc_usdt_fixture
from liq.examples.models.baseline import buy_and_hold
from liq.examples.models.linear import LinearSignalModel
from liq.examples.pipelines.pipeline import fit_pipeline
from liq.features.pipeline import FeaturePipeline


def test_pipeline_and_models_smoke() -> None:
    df = btc_usdt_fixture()
    pipeline = fit_pipeline(df)
    # baseline orders
    bh_orders = buy_and_hold(df, symbol="BTC_USDT")
    assert bh_orders

    # model fit/predict
    model = LinearSignalModel().fit(df)
    orders = model.predict(df, symbol="BTC_USDT")
    # may be zero depending on data; just ensure no crash
    assert orders is not None

    # ensure pipeline can transform returns
    mid = (df["high"] + df["low"]) / 2
    rets = mid.pct_change().fill_null(0).to_list()
    transformed = pipeline.transform(rets)
    assert len(transformed) == len(rets)
