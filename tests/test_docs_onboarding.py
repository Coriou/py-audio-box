"""Keep user-facing onboarding docs tied to the code they describe.

Checks: relative links/anchors resolve; every documented ``clients.cli`` command
parses and maps to a contract operation; every documented ``speech_*`` MCP tool
exists. No network, Docker or service is used.
"""
import contextlib
import io
import json
import re
import shlex
import unittest
import tempfile
import zipfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
# Every link target here must also be copied by Dockerfile.tests.
DOCS = ['README.md', 'docs/AGENT-GUIDE.md', 'docs/LEGACY-TOOLBOX.md', 'docs/agent-interface/CPU-QUICKSTART.md',
        'docs/agent-interface/POCKET-QUICKSTART.md', 'docs/agent-interface/HEADLESS-INSTALL.md']
CLI_DOCS = DOCS
EXTRA_MCP_TOOLS = {'speech_asset_upload_file', 'speech_status', 'speech_skeleton_status', 'speech_operation_detail'}


def slug(heading):
    text = re.sub(r'[`*_]|\[([^\]]*)\]\([^)]*\)', r'\1', heading.strip().lower())
    return re.sub(r'\s', '-', re.sub(r'[^\w\s-]', '', text))


def anchors(path):
    found, fenced = set(), False
    for line in path.read_text().splitlines():
        if line.startswith('```'): fenced = not fenced
        elif not fenced and (m := re.match(r'#{1,6}\s+(.*)', line)): found.add(slug(m.group(1)))
    return found


def commands(text):
    """Yield argv lists for documented `python -m clients.cli ...` lines."""
    joined = re.sub(r'\\\n\s*', ' ', text)
    for block in re.findall(r'```(?:sh|bash)?\n(.*?)```', joined, re.S):
        block = re.sub(r"'\{[^']*\}'", "'{}'", block)  # placeholder JSON bodies
        for line in block.splitlines():
            if 'clients.cli' not in line: continue
            argv = shlex.split(line.split('#', 1)[0])
            argv = argv[argv.index('clients.cli') + 1:]
            if argv and argv != ['--help']: yield argv


class DocsOnboarding(unittest.TestCase):
    def test_agent_commands_use_client_python_and_canonical_status(self):
        guide = (ROOT/'docs/AGENT-GUIDE.md').read_text()
        self.assertNotRegex(guide, r'(?<![\w/.-])python -m clients\.cli')
        self.assertIn('.venv-client/bin/python -m clients.cli status', guide)
        self.assertIn('speech_status', guide)
        self.assertIn('deprecated aliases', guide)
        self.assertIn('text-fidelity failures', guide)

    def test_pocket_recipe_pins_and_checks_descriptor(self):
        doc = (ROOT/'docs/agent-interface/POCKET-QUICKSTART.md').read_text()
        descriptor = json.loads((ROOT/'requirements/tts/pocket-fr.expected.json').read_text())
        for name, size, digest in (
            ('model.safetensors', 219029196, 'dd5fe98f5577ecfd56db9af8057cd05fe20cbc91560042aec8a56cfb01c34995'),
            ('tokenizer.json', 246403, '6211acffe9cb95172f71f0051f18e26ea305b2b32ea00b331ee976551c70887f')):
            self.assertIn(str(size), doc)
            self.assertIn(digest, doc)
            self.assertIn(digest, json.dumps(descriptor))
            self.assertIn('fresh-pocket-cache/' + name, doc)
        self.assertIn('set -eu', doc)
        self.assertIn('hf auth login', doc)
        self.assertIn('--revision d0ebc5deb50bc6b225f686782f124dd37e2dca0a', doc)
        self.assertLess(doc.index('mkdir -m 700'), doc.index('mv /absolute/private'))
        self.assertIn("hashlib.file_digest(source, 'sha256')", doc)
        self.assertIn('config.yaml', doc)

    def test_relative_links_and_anchors_resolve(self):
        broken = []
        for name in DOCS:
            doc = ROOT / name
            for target in re.findall(r'\]\(([^)\s]+)\)', doc.read_text()):
                if re.match(r'[a-z]+:', target): continue
                file, _, anchor = target.partition('#')
                path = (doc.parent / file).resolve() if file else doc
                if not path.exists(): broken.append(f'{name}: {target}')
                elif anchor and path.suffix == '.md' and anchor not in anchors(path):
                    broken.append(f'{name}: {target} (anchor)')
        self.assertEqual(broken, [])

    def test_archive_user_documentation_closure(self):
        files = set(json.loads((ROOT / 'headless-dist-manifest.json').read_text())['files'])
        self.assertTrue({'README.md', 'docs/AGENT-GUIDE.md'} <= files)
        from service.package_install import archive
        with tempfile.TemporaryDirectory(prefix='pab-docs-archive-') as tmp:
            destination = Path(tmp) / 'docs.zip'
            archive(ROOT, destination)
            with zipfile.ZipFile(destination) as package:
                for name in files:
                    self.assertEqual(package.read('speech-box/' + name), (ROOT / name).read_bytes(), name)
        # Walk user docs, not the optional maintainer proof reports already shipped.
        pending = ['README.md', 'docs/AGENT-GUIDE.md']
        visited = set()
        while pending:
            name = pending.pop()
            if name in visited: continue
            visited.add(name)
            doc = ROOT / name
            for target in re.findall(r'\]\(([^)\s]+)\)', doc.read_text()):
                if re.match(r'[a-z]+:', target): continue
                file, _, anchor = target.partition('#')
                path = (doc.parent / file).resolve() if file else doc
                relative = path.relative_to(ROOT).as_posix()
                with self.subTest(doc=name, target=target):
                    self.assertIn(relative, files, 'required linked file missing from archive')
                    self.assertTrue(path.is_file())
                    if anchor and path.suffix == '.md': self.assertIn(anchor, anchors(path))
                if path.suffix == '.md' and '/reviews/' not in relative:
                    pending.append(relative)

    def test_documented_cli_commands_parse_and_name_contract_operations(self):
        import clients.cli as cli
        operations = {o['id'] for o in json.loads((ROOT / 'contracts/v1/operations.json').read_text())['operations']}
        for name in CLI_DOCS:
            for argv in commands((ROOT / name).read_text()):
                with self.subTest(doc=name, argv=argv):
                    called = []
                    fake = mock.MagicMock()
                    fake.call.side_effect = lambda op, *a, **k: called.append(op) or {}
                    fake.get.side_effect = lambda path, *a, **k: called.append(path) or {}
                    fake.upload_file.side_effect = lambda *a, **k: called.append('asset_upload') or {}
                    out = io.StringIO()
                    with mock.patch.object(cli, 'HTTP', return_value=fake), contextlib.redirect_stdout(out), \
                            contextlib.redirect_stderr(io.StringIO()):
                        try: code = cli.main(argv)
                        except SystemExit as exc: code = exc.code
                        except (KeyError, TypeError): code = None  # stub lacks byte metadata; dispatch already checked
                    self.assertTrue(called, f'command did not reach the service: {argv} -> {code} {out.getvalue()}')
                    for op in called:
                        self.assertTrue(op.startswith('/') or op in operations, op)

    def test_documented_mcp_tools_and_operation_ids_exist(self):
        ops = json.loads((ROOT / 'contracts/v1/operations.json').read_text())['operations']
        tools = {o['mcp'] for o in ops} | {o['id'] for o in ops} | EXTRA_MCP_TOOLS
        for name in DOCS:
            for tool in set(re.findall(r'`(speech_[a-z_]+)', (ROOT / name).read_text())):
                with self.subTest(doc=name, tool=tool):
                    self.assertIn(tool, tools)


if __name__ == '__main__':
    unittest.main()
