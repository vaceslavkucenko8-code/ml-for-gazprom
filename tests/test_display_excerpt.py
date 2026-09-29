from src.display_excerpt import select_excerpt


def test_navigation_not_used_as_technology_description():
    text = ('Official websites use secure websites and privacy policy information.\n'
            'Subscribe to receive all updates about quantum sensor technology.\n'
            'The quantum sensor prototype measures weak magnetic fields during laboratory experiments.')
    spans = select_excerpt(text, 'Quantum sensor prototype')
    assert len(spans) == 1
    assert spans[0]['quote'].startswith('The quantum sensor')
    assert all(text[s['start']:s['end']] == s['quote'] for s in spans)


def test_no_relevant_text_means_no_invented_description():
    assert select_excerpt('The robotic actuator performs industrial manipulation with a flexible gripper.', 'Quantum sensor') == []
