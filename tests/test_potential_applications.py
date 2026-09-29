from src.extraction.candidates import is_future_looking


def test_proposed_application_is_not_observed():
    assert is_future_looking('This device could be used for high-speed imaging.')
    assert is_future_looking('It may improve throughput.')
    assert not is_future_looking('The researchers demonstrated image classification.')
    assert not is_future_looking('The device includes a microwave circuit.')
