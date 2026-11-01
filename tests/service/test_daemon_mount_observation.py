"""Source-only observation bracket tests; no Docker or installed runtime."""
import copy
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

from service import daemon_helper as dh, daemon_volumes as dv
from service import installed_qwen_daemon as qd
from tests.service.daemon_volume_fixtures import binding, container
from tests.service import test_split_init as split_fixtures
from tests.service import test_installed_qwen as qwen_fixtures


class MountObservationTests(unittest.TestCase):
    def setUp(self):
        fixture=split_fixtures.SplitInit()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.config=fixture.config
        self.family=None
        self.binding=binding(self.config)
        self.first={role:container(role,self.config,self.binding) for role in ('control','worker')}
        for c in self.first.values():
            c['State'].update(Running=True,Status='running',StartedAt='2026-10-25T00:00:01Z')
        self.ids={role:c['Id'] for role,c in self.first.items()}
        self.startup=dict(hello=dict(boot='b'*32),local=dict(roots={}))

    def observe(self, second, later=None, *, assets=None, asset_view=None):
        # Real Helper.observe -> pair -> inspect -> daemon_volumes.inspect_container.
        helper=object.__new__(dh.Helper)
        helper.config=self.config
        helper.b=self.binding
        helper.family=self.family
        helper.evidence=dv.EVIDENCE
        helper.export=False
        helper.pocket=assets is not None
        if assets is not None:
            helper.root={'control':Path('/inert-control')}
        helper.read=lambda _:dict(identities={})
        counts=dict(control=0,worker=0)
        def rpc(action, request):
            if action=='facts':return dict(roots={'assets':assets} if assets is not None else {})
            self.assertEqual(action,'inspect')
            role=request['role']
            count=counts[role]
            counts[role]+=1
            return copy.deepcopy((self.first if count==0 else second)[role])
        helper.rpc=rpc
        startup=copy.deepcopy(self.startup)
        if assets is not None: startup['local']['roots']['assets']=assets
        startups=iter((startup,startup if later is None else later))
        helper.startup=lambda:copy.deepcopy(next(startups))
        with patch.object(dh.ip,'directory_identity',return_value=tuple(assets if asset_view is None else asset_view)) if assets is not None else nullcontext():
            result=helper.observe(self.ids,{})
        self.assertEqual(counts,dict(control=2,worker=2))
        return result

    def test_both_mount_lists_permute_without_changing_evidence_or_inputs(self):
        second=copy.deepcopy(self.first)
        for role in ('control','worker'):
            second[role]['Mounts'].reverse()
            second[role]['HostConfig']['Mounts'].reverse()
        original=copy.deepcopy(second)
        result=self.observe(second)
        self.assertEqual(result['inspections'],self.first)
        self.assertEqual(second,original)
        # Registration and notification inspect again after the first bracket.
        later=copy.deepcopy(result)
        later['inspections']=second
        self.assertEqual(dh.comparison_observation(result),dh.comparison_observation(later))
        later['hello']['boot']='c'*32
        self.assertNotEqual(dh.comparison_observation(result),dh.comparison_observation(later))

    def test_qwen_mount_lists_permute_through_real_family_validator(self):
        fixture=qwen_fixtures.InstalledQwenTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.config=fixture.config
        self.family=qd.FAMILY
        self.binding=binding(self.config)
        self.binding['policy']=qd.LAUNCH
        self.first={role:container(role,self.config,self.binding,family=self.family)
                    for role in ('control','worker')}
        for c in self.first.values():
            c['State'].update(Running=True,Status='running',StartedAt='2026-10-25T00:00:01Z')
        self.ids={role:c['Id'] for role,c in self.first.items()}
        second=copy.deepcopy(self.first)
        for c in second.values():
            c['Mounts'].reverse()
            c['HostConfig']['Mounts'].reverse()
        self.assertEqual(self.observe(second)['inspections'],self.first)

    def test_resident_qwen_seven_mount_worker_permutations_and_custody_refusals(self):
        from tests.service.test_qwen_service_path import ServicePathTests
        fixture=ServicePathTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        self.config=fixture.config
        self.family=qd.FAMILY
        self.binding=binding(self.config)
        self.binding.update(policy=qd.LAUNCH,custody='/sys/fs/cgroup/speech-box-qwen/'+'b'*32)
        self.first={role:container(role,self.config,self.binding,family=self.family)
                    for role in ('control','worker')}
        for c in self.first.values():
            c['State'].update(Running=True,Status='running',StartedAt='2026-10-25T00:00:01Z')
        self.ids={role:c['Id'] for role,c in self.first.items()}
        self.assertEqual(len(self.first['worker']['Mounts']),7)
        self.assertEqual(len(self.first['worker']['HostConfig']['Mounts']),7)
        assets=[101,202]
        for rotation in range(10):
            with self.subTest(rotation=rotation):
                second=copy.deepcopy(self.first)
                for role in ('control','worker'):
                    mounts=second[role]['Mounts']
                    offset=rotation%len(mounts)
                    mounts[:]=mounts[offset:]+mounts[:offset]
                    mounts=second[role]['HostConfig']['Mounts']
                    offset=(rotation+1)%len(mounts)
                    mounts[:]=mounts[offset:]+mounts[:offset]
                original=copy.deepcopy(second)
                result=self.observe(second,assets=assets)
                self.assertEqual(result['inspections'],self.first)
                self.assertEqual(second,original)
                self.assertEqual(result['local']['control']['roots']['assets'],assets)
        for change in ('source','target','readonly','propagation','missing'):
            with self.subTest(custody=change):
                second=copy.deepcopy(self.first)
                requested=next(m for m in second['worker']['HostConfig']['Mounts'] if m['Type']=='bind')
                if change=='source': requested['Source']='/wrong'
                elif change=='target': requested['Target']='/wrong'
                elif change=='readonly': requested['ReadOnly']=True
                elif change=='propagation': requested['BindOptions']['Propagation']='private'
                else: second['worker']['Mounts'].pop()
                with self.assertRaises(ValueError): self.observe(second,assets=assets)
        with self.assertRaisesRegex(ValueError,'daemon_assets_view_identity'):
            self.observe(copy.deepcopy(self.first),assets=assets,asset_view=[101,203])

    def test_real_mount_changes_and_duplicates_are_rejected(self):
        for role in ('control','worker'):
            for change in ('source','destination','rw','missing','extra','duplicate',
                           'requested_source','requested_missing','requested_duplicate','extra_field'):
                with self.subTest(role=role,change=change):
                    second=copy.deepcopy(self.first)
                    c=second[role]
                    if change=='source':c['Mounts'][0]['Source']='/different'
                    elif change=='destination':c['Mounts'][0]['Destination']='/different'
                    elif change=='rw':c['Mounts'][0]['RW']=not c['Mounts'][0]['RW']
                    elif change=='missing':c['Mounts'].pop()
                    elif change=='extra':c['Mounts'].append(dict(c['Mounts'][0],Destination='/extra'))
                    elif change=='duplicate':c['Mounts'].append(copy.deepcopy(c['Mounts'][0]))
                    elif change=='requested_source':c['HostConfig']['Mounts'][0]['Source']='different'
                    elif change=='requested_missing':c['HostConfig']['Mounts'].pop()
                    elif change=='requested_duplicate':c['HostConfig']['Mounts'].append(copy.deepcopy(c['HostConfig']['Mounts'][0]))
                    else:c['Mounts'][0]['Unexpected']='changed'
                    with self.assertRaises(ValueError):self.observe(second)

    def test_identity_image_lifecycle_security_and_startup_changes_rejected(self):
        for field in ('Id','Image','StartedAt','Running','Privileged','other'):
            with self.subTest(field=field):
                second=copy.deepcopy(self.first)
                c=second['worker']
                if field=='Id':c['Id']='f'*64
                elif field=='Image':c['Image']='sha256:'+'f'*64
                elif field=='StartedAt':c['State']['StartedAt']='2026-10-25T00:00:02Z'
                elif field=='Running':c['State']['Running']=False
                elif field=='Privileged':c['HostConfig']['Privileged']=True
                else:c['NetworkSettings']={'new':'value'}
                with self.assertRaises(ValueError):self.observe(second)
        later=copy.deepcopy(self.startup)
        later['hello']['boot']='c'*32
        with self.assertRaisesRegex(ValueError,'daemon_observation_startup_changed'):
            self.observe(copy.deepcopy(self.first),later)


if __name__=='__main__':unittest.main()
