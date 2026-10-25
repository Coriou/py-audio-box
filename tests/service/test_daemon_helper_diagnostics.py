import pytest

from service import daemon_helper


def test_refusal_location_does_not_disclose_exception_payload(monkeypatch, capsys):
    monkeypatch.setattr(daemon_helper.sys, 'platform', 'linux')
    def denied():
        raise ValueError('private-credential-or-manifest-content')
    monkeypatch.setattr(daemon_helper.os, 'getresuid', denied, raising=False)
    with pytest.raises(SystemExit) as stopped:
        daemon_helper.main()
    assert stopped.value.code == 2
    output = capsys.readouterr()
    assert output.out == ''
    assert output.err.startswith('daemon_helper_refused:ValueError:denied:')
    assert 'private-credential-or-manifest-content' not in output.err
    assert len(output.err) < 128
