from gb_regressor import pipeline
from gb_regressor.config.core import config


def test_pipeline_drops_unnecessary_features(pipeline_inputs):
    # Given
    X_train, X_test, y_train, y_test = pipeline_inputs
    assert config.model_config.drop_features in X_train.columns
    pipeline.price_pipe.fit(X_train, y_train)

    # When
    # We access the transformed inputs with slicing
    transformed_inputs = pipeline.price_pipe[:-1].transform(X_train)

    # Then
    assert config.model_config.drop_features in X_train.columns
    assert config.model_config.drop_features not in transformed_inputs.columns


def test_pipeline_transforms_temporal_features(pipeline_inputs):
    # Given
    X_train, X_test, y_train, y_test = pipeline_inputs

    # When
    # We access the transformed inputs with slicing
    transformed_inputs = pipeline.price_pipe[:-1].transform(X_train)

    # Then
    assert (transformed_inputs["YearRemodAdd"] == (X_train["YrSold"] - X_train["YearRemodAdd"])).all()
