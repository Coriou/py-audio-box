"""CPU-only policy fixtures; no hardware qualification."""
import unittest
from dataclasses import replace
from service.profiles import (ExecutionProfile, Resources, Performance, VoiceWorkRequest,
                              SelectionPolicy, Reason, select)
from worker.hardware import DeviceFamily as F, Host, Device, discover


def profile(name='cpu', family=F.CPU, **changes):
    p = ExecutionProfile(name, 'qwen', 'checkpoint', 'immutable-revision', 'reference-sha',
                         family, 0, 'float32', 'eager', frozenset(('clone', 'design', 'preset')),
                         frozenset(('en', 'fr')), Resources(1024, 2, 1), True, True, True,
                         '2026-10-24', 'fake-qualification', Performance('fixture', 8, 2, 'fake-measurement'))
    return replace(p, **changes)


def request(**changes):
    return replace(VoiceWorkRequest('qwen', 'checkpoint', 'immutable-revision', 'reference-sha',
                                   frozenset(('clone',)), 'fr', 8192), **changes)


def inventory(system='linux', arch='x86_64', deployment='docker'):
    return discover(Host(system, arch, deployment), tuple(
        Device(f, 0, True, True, 'fake-probe') for f in F))


class SelectionTests(unittest.TestCase):
    def choose(self, profiles, r=None, inv=None, policy=None):
        return select(tuple(profiles), r or request(), inv or inventory(),
                      policy or SelectionPolicy(workload='fixture'))

    def test_cpu_constrained_and_missing_backend(self):
        self.assertEqual(self.choose([profile()]).profile.resources.concurrency, 1)
        result = self.choose([profile(installed=False)])
        self.assertEqual(result.status, 'not_ready')
        self.assertIn(Reason.BACKEND, result.diagnostics[0].reasons)

    def test_accelerator_measured_compatible_and_no_cpu_cap(self):
        gpu = profile('gpu', F.CUDA, resources=Resources(4096, 8, 4),
                      performance=Performance('fixture', 3, .5, 'fake'))
        self.assertEqual(self.choose([profile(), gpu], request(concurrency=3)).profile, gpu)
        self.assertEqual(self.choose([profile(), gpu]).profile, gpu)

    def test_native_mps_and_docker(self):
        mps = profile('metal', F.MPS, dtype='float16')
        self.assertEqual(self.choose([mps], inv=inventory('darwin', 'arm64', 'native')).profile, mps)
        for host in (inventory('linux', 'aarch64'), inventory('darwin', 'arm64', 'docker'),
                     inventory('darwin', 'arm64', 'unknown')):
            self.assertEqual(self.choose([mps], inv=host).status, 'not_ready')
            self.assertEqual(self.choose([profile()], inv=host).status, 'ready')

    def test_identity_features_language_execution(self):
        for field, value, reason in (
                ('engine', 'kokoro', Reason.IDENTITY), ('checkpoint', 'other', Reason.IDENTITY),
                ('revision', 'other', Reason.IDENTITY), ('conditioning', 'other', Reason.IDENTITY),
                ('features', frozenset(('unsupported',)), Reason.FEATURES),
                ('language', 'de', Reason.LANGUAGE), ('dtype', 'bfloat16', Reason.EXECUTION),
                ('attention', 'flash', Reason.EXECUTION)):
            with self.subTest(field=field):
                result = self.choose([profile()], request(**{field: value}))
                self.assertEqual(result.status, 'not_ready')
                self.assertIn(reason, result.diagnostics[0].reasons)

    def test_readiness_gates(self):
        for change, reason in (({'qualified_at': None}, Reason.QUALIFICATION),
                               ({'qualification_evidence': None}, Reason.QUALIFICATION),
                               ({'provisioned': False}, Reason.PROVISION),
                               ({'verified': False}, Reason.PROVISION)):
            self.assertIn(reason, self.choose([profile(**change)]).diagnostics[0].reasons)
        for device in (Device(F.CPU, 0, True, False, 'fake'),
                       Device(F.CPU, 0, False, True, 'fake'),
                       Device(F.CPU, 0, True, True, None)):
            self.assertEqual(self.choose([profile()], inv=discover(Host('linux', 'x86_64', 'native'), (device,))).status, 'not_ready')

    def test_memory_concurrency(self):
        for r, reason in ((request(memory_available_bytes=512), Reason.MEMORY),
                          (request(concurrency=2), Reason.CONCURRENCY)):
            self.assertIn(reason, self.choose([profile()], r).diagnostics[0].reasons)

    def test_overrides_never_fall_back(self):
        policy = SelectionPolicy(workload='fixture', preferred_device=F.CUDA, fallback_devices=(F.CPU,))
        for r in (request(profile_override='missing'), request(device_override=(F.CUDA, 0)),
                  request(profile_override='cpu', device_override=(F.CUDA, 0))):
            result = self.choose([profile()], r, policy=policy)
            self.assertEqual(result.reason, Reason.OVERRIDE_UNAVAILABLE)
        self.assertEqual(self.choose([profile()], request(profile_override='cpu')).reason, Reason.SELECTED_OVERRIDE)

    def test_explicit_fallback_identity_only(self):
        self.assertEqual(self.choose([profile()], policy=SelectionPolicy(
            workload='fixture', preferred_device=F.CUDA)).status, 'not_ready')
        policy = SelectionPolicy(workload='fixture', preferred_device=F.CUDA, fallback_devices=(F.CPU,))
        self.assertEqual(self.choose([profile()], policy=policy).reason, Reason.EXPLICIT_FALLBACK)
        self.assertEqual(self.choose([profile(revision='different')], policy=policy).status, 'not_ready')

    def test_objective_and_deterministic_unmeasured(self):
        a = profile('a', performance=Performance('fixture', 1, 9, 'fake'))
        b = profile('b', performance=Performance('fixture', 9, 1, 'fake'))
        self.assertEqual(self.choose([a, b]).profile, b)
        self.assertEqual(self.choose([b, a], policy=SelectionPolicy('cold', 'fixture')).profile, a)
        a, b = replace(a, performance=None), replace(b, performance=None)
        self.assertEqual(self.choose([a, b]).reason, Reason.MEASUREMENT)
        for rows in ([a, b], [b, a]):
            result = self.choose(rows, policy=SelectionPolicy(allow_unmeasured=True))
            self.assertEqual(result.profile.profile_id, 'a')
            self.assertEqual(result.reason, Reason.SELECTED_UNMEASURED)
        self.assertEqual(self.choose([profile()], policy=SelectionPolicy(workload='different')).status, 'not_ready')

    def test_invalid_and_bounded_records(self):
        for v in (0, -1, True):
            with self.assertRaises(ValueError):
                Resources(v, 1, 1)
        with self.assertRaises(ValueError):
            Performance('fixture', float('nan'), 1, 'fake')
        with self.assertRaises(ValueError):
            self.choose([profile(), profile()])
        with self.assertRaises(ValueError):
            self.choose([profile(str(i)) for i in range(257)])


if __name__ == '__main__':
    unittest.main()
