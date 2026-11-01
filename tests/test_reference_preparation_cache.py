"""Preparation control-flow tests using synthetic WAVs and injected ML/codec boundaries.

Compile the actual prepare_ref and quality functions from the app AST so these
checks require no model dependencies or downloads. A codec check uses existing
ffmpeg/ffprobe when available. This does not qualify VAD/ASR accuracy, complete app imports, synthesis or Linux runtime readiness.
"""
import ast
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import Any
import unittest
import wave

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / 'apps/voice-clone/voice-clone.py'


def duration(path):
    with wave.open(str(path), 'rb') as audio:
        return audio.getnframes() / audio.getframerate()


class ReferencePreparationCache(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source.wav'
        # Every second has distinct PCM bytes, making bounds mistakes observable.
        with wave.open(str(self.source), 'wb') as audio:
            audio.setparams((1, 2, 100, 0, 'NONE', 'not compressed'))
            audio.writeframes(b''.join(struct.pack('<h', second * 100) * 100
                                       for second in range(24)))
        self.calls = dict(normalize=0, trim=0, transcribe=0, select=0)
        self.bounds = (2.0, 8.0)

        def normalize(src, dest):
            self.calls['normalize'] += 1
            shutil.copyfile(src, dest)

        def trim(src, start, length, dest):
            self.calls['trim'] += 1
            with wave.open(str(src), 'rb') as audio:
                params = audio.getparams()
                audio.setpos(round(start * audio.getframerate()))
                frames = audio.readframes(round(length * audio.getframerate()))
            with wave.open(str(dest), 'wb') as audio:
                audio.setparams(params)
                audio.writeframes(frames)

        def transcribe(path, model, threads):
            self.calls['transcribe'] += 1
            return 'Synthetic speech ' + hashlib.sha256(path.read_bytes()).hexdigest(), -.3, 'English', .99

        def select(candidates, normalized, directory, model, threads, force, interactive):
            self.calls['select'] += 1
            start, end = self.bounds
            (directory / 'best_segment.json').write_text(json.dumps(dict(
                seg_start=start, seg_end=end, candidate_scoring_version=2)))
            return (start, end), '', 0., 'English', 1.

        self.ns = dict(Path=Path, Any=Any, math=math, json=json, sys=sys,
            datetime=datetime, timezone=timezone, RefResult=SimpleNamespace,
            Timer=lambda label: nullcontext(), CANDIDATE_SCORING_VERSION=2,
            QUALITY_GATE_LOGPROB=-1.2, QUALITY_GATE_LANGUAGE_PROBABILITY=.8,
            REFERENCE_QUALITY_POLICY_VERSION=1, MAX_REF_SECONDS=12.,
            sha256_file=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()[:32],
            normalize_audio=normalize, trim_audio_encode=trim, get_duration=duration,
            transcribe_ref=transcribe,
            atomic_json=lambda path, data: path.write_text(json.dumps(data, allow_nan=False)),
            load_silero=lambda **kw: (None, None),
            _pick_top_candidates=lambda *a: [self.bounds],
            _score_and_select_best_candidate=select)
        names = {'prepare_ref', '_finite_metric', 'reference_quality_report',
                 'candidate_scoring_cache_is_current'}
        nodes = [node for node in ast.parse(APP.read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(APP), 'exec'), self.ns)
        self.kwargs = dict(ref_audio=self.source, cache=self.root / 'cache', whisper_model='small',
                           num_threads=1, ref_start=0., ref_end=6., ref_language='Auto',
                           x_vector_only=False, force=False)

    def prepare(self, **kwargs):
        return self.ns['prepare_ref'](**{**self.kwargs, **kwargs})

    def test_changed_manual_bounds_keep_old_audio_transcript_and_metadata(self):
        first = self.prepare()
        before = {p: p.read_bytes() for p in first.ref_dir.iterdir()}
        second = self.prepare(ref_start=12., ref_end=20.)
        self.assertNotEqual(first.ref_dir, second.ref_dir)
        self.assertNotEqual(first.segment.read_bytes(), second.segment.read_bytes())
        self.assertNotEqual(first.transcript, second.transcript)
        self.assertEqual(duration(second.segment), 8.)
        with wave.open(str(second.segment), 'rb') as audio:
            self.assertEqual(struct.unpack('<h', audio.readframes(1))[0], 1200)
        for p, data in before.items():
            self.assertEqual(p.read_bytes(), data)
        for name in ('pipeline_state.json', 'ref_transcript.json'):
            state = json.loads((second.ref_dir / name).read_text())
            self.assertEqual((state['seg_start'], state['seg_end']), (12., 20.))
            self.assertEqual(state['transcript'], second.transcript)
            self.assertEqual(state['preparation']['selection'], 'manual')
        calls = self.calls.copy()
        again = self.prepare(ref_start=12, ref_end=20, num_threads=4)
        self.assertEqual(second.ref_dir, again.ref_dir)
        self.assertEqual(again.transcript, second.transcript)
        self.assertEqual(calls, self.calls)

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),
                         'requires existing ffmpeg and ffprobe; never installs them')
    def test_real_codec_manual_bounds_and_identical_reuse(self):
        # Exercise the actual repository codec helpers with synthetic tones.
        audio_source = ROOT / 'lib/audio.py'
        names = {'normalize_audio', 'trim_audio_encode', 'get_duration'}
        nodes = [n for n in ast.parse(audio_source.read_text()).body
                 if isinstance(n, ast.FunctionDef) and n.name in names]
        helpers = dict(Path=Path, subprocess=subprocess, json=json)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(audio_source), 'exec'), helpers)
        for name in names:
            self.ns[name] = helpers[name]
        with wave.open(str(self.source), 'wb') as audio:
            audio.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
            audio.writeframes(b''.join(struct.pack('<h', int(6000 * math.sin(
                2 * math.pi * (220 if i < 12 * 16000 else 440) * i / 16000)))
                for i in range(24 * 16000)))
        first = self.prepare()
        original = first.segment.read_bytes()
        second = self.prepare(ref_start=12., ref_end=20.)
        self.assertNotEqual(original, second.segment.read_bytes())
        self.assertEqual(duration(first.segment), 6.)
        self.assertEqual(duration(second.segment), 8.)
        self.assertNotEqual(first.transcript, second.transcript)
        self.assertEqual(first.segment.read_bytes(), original)
        calls = self.calls['transcribe']
        # A cache hit must not invoke either codec helper again.
        def unexpected(*a, **kw):
            self.fail('identical preparation must reuse normalized and segment WAVs')
        self.ns['normalize_audio'] = self.ns['trim_audio_encode'] = unexpected
        again = self.prepare(ref_start=12., ref_end=20.)
        self.assertEqual(again.ref_dir, second.ref_dir)
        self.assertEqual(again.transcript, second.transcript)
        self.assertEqual(self.calls['transcribe'], calls)

    def test_legacy_source_only_cache_is_preserved_and_never_used(self):
        legacy = self.kwargs['cache'] / 'voice-clone/refs' / self.ns['sha256_file'](self.source)
        legacy.mkdir(parents=True)
        old = {name: b'old unidentified artifact' for name in
               ('ref_segment.wav', 'ref_transcript.json', 'best_segment.json', 'pipeline_state.json')}
        for name, data in old.items():
            (legacy / name).write_bytes(data)
        result = self.prepare()
        self.assertNotEqual(result.ref_dir, legacy)
        self.assertEqual(self.calls['transcribe'], 1)
        for name, data in old.items():
            self.assertEqual((legacy / name).read_bytes(), data)

    def test_auto_and_manual_are_distinct_and_auto_reuses(self):
        manual = self.prepare(ref_start=2., ref_end=8.)
        auto = self.prepare(ref_start=None, ref_end=None)
        self.assertNotEqual(manual.ref_dir, auto.ref_dir)
        self.assertEqual(manual.segment.read_bytes(), auto.segment.read_bytes())
        self.assertEqual(self.calls['select'], 1)
        before = self.calls.copy()
        again = self.prepare(ref_start=None, ref_end=None)
        self.assertEqual(auto.ref_dir, again.ref_dir)
        self.assertEqual(self.calls, before)

    def test_automatic_selection_is_invalidated_by_whisper_and_policy(self):
        first = self.prepare(ref_start=None, ref_end=None)
        self.bounds = (12., 20.)
        second = self.prepare(ref_start=None, ref_end=None, whisper_model='medium')
        self.assertNotEqual(first.ref_dir, second.ref_dir)
        self.assertEqual((second.seg_start, second.seg_end), self.bounds)
        self.ns['CANDIDATE_SCORING_VERSION'] = 3
        third = self.prepare(ref_start=None, ref_end=None)
        self.assertNotEqual(first.ref_dir, third.ref_dir)
        self.assertEqual(self.calls['select'], 3)

    def test_interactive_reselection_keeps_previous_pick(self):
        first = self.prepare(ref_start=None, ref_end=None, interactive=True)
        data = first.segment.read_bytes()
        self.bounds = (12., 20.)
        second = self.prepare(ref_start=None, ref_end=None, interactive=True)
        self.assertNotEqual(first.ref_dir, second.ref_dir)
        self.assertNotEqual(first.transcript, second.transcript)
        self.assertEqual(first.segment.read_bytes(), data)
        self.assertEqual(self.calls['select'], 2)

    def test_force_recomputes_normalization_selection_trim_and_asr(self):
        first = self.prepare(ref_start=None, ref_end=None)
        counts = self.calls.copy()
        second = self.prepare(ref_start=None, ref_end=None, force=True)
        self.assertEqual(first.ref_dir, second.ref_dir)
        self.assertEqual(self.calls, {key: count + 1 for key, count in counts.items()})

    def test_preparation_options_isolate_asr_and_explicit_bypasses(self):
        first = self.prepare()
        original = (first.ref_dir / 'ref_transcript.json').read_bytes()
        for options in ({'whisper_model': 'medium'}, {'ref_language': 'French'},
                        {'ref_text': 'Provided words'}, {'ref_text': 'Different words'},
                        {'x_vector_only': True}):
            with self.subTest(options=options):
                result = self.prepare(**options)
                self.assertNotEqual(result.ref_dir, first.ref_dir)
                source = json.loads((result.ref_dir / 'ref_transcript.json').read_text())['transcript_source']
                self.assertEqual(source, 'provided_ref_text' if 'ref_text' in options else
                                 'x_vector_only' if options.get('x_vector_only') else 'asr')
        self.assertEqual((first.ref_dir / 'ref_transcript.json').read_bytes(), original)
        calls = self.calls.copy()
        self.assertEqual(self.prepare().ref_dir, first.ref_dir)
        self.assertEqual(self.calls, calls)

    def test_cached_quality_is_rechecked_without_implicit_override(self):
        self.ns['transcribe_ref'] = lambda *a: ('Dubious speech', -.7, 'English', .63)
        result = self.prepare(force_bad_ref=True)
        with self.assertRaises(SystemExit):
            self.prepare()
        record = json.loads((result.ref_dir / 'ref_transcript.json').read_text())
        self.assertFalse(record['quality_report']['accepted'])
        self.assertFalse(record['quality_report']['override'])

    def test_invalid_manual_bounds_rejected_without_trim_or_transcription(self):
        for start, end in ((None, 6.), (0., None), (-1., 4.), (4., 4.), (8., 4.),
                           (float('nan'), 4.), (0., float('inf')), (0., 25.)):
            with self.subTest(bounds=(start, end)), self.assertRaises(ValueError):
                self.prepare(ref_start=start, ref_end=end)
        self.assertEqual(self.calls['trim'], 0)
        self.assertEqual(self.calls['transcribe'], 0)
        with self.assertRaises(ValueError):
            self.prepare(ref_text='  ')


if __name__ == '__main__':
    unittest.main()
