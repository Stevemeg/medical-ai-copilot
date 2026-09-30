"""Scanner coverage and ephemeral CI key handling must fail safely."""

import json
from types import SimpleNamespace

from scripts import ci_test_keys, security_audit


def test_cpu_build_mapping_is_exact_and_unknown_builds_remain_unmapped():
    assert security_audit.audit_version("torch", "2.14.0+cpu") == "2.14.0"
    assert security_audit.audit_version("torch", "2.14.0+unknown") == "2.14.0+unknown"
    assert security_audit.audit_version("other", "2.14.0+cpu") == "2.14.0+cpu"


def test_audit_does_not_accept_skipped_distribution(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "security_exceptions.json").write_text(json.dumps({"exceptions": []}))
    monkeypatch.setattr(
        security_audit, "distributions", lambda: [SimpleNamespace(metadata={"Name": "torch"}, version="2.14.0+cpu")]
    )

    def scanner(command, check):
        assert "--disable-pip" in command and "--no-deps" in command
        output = command[command.index("--output") + 1]
        security_audit.Path(output).write_text(
            json.dumps({"dependencies": [{"name": "torch", "skip_reason": "Unknown release"}]})
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(security_audit.subprocess, "run", scanner)
    assert security_audit.main() == 1
    report = json.loads((tmp_path / "reports/security-summary.json").read_text())
    assert report["unresolved"] and report["audited_distributions"] == 1
    assert report["build_version_mappings"] == [
        {"package": "torch", "installed": "2.14.0+cpu", "audited_upstream": "2.14.0"}
    ]


def test_ci_keys_are_independent_and_masked(tmp_path, monkeypatch, capsys):
    target = tmp_path / "environment"
    monkeypatch.setenv("GITHUB_ENV", str(target))
    ci_test_keys.main()
    lines = target.read_text().splitlines()
    values = [line.split("=", 1)[1] for line in lines]
    assert len(set(values)) == 2 and all(len(v) == 64 for v in values)
    assert capsys.readouterr().out.splitlines() == ["::add-mask::" + value for value in values]
