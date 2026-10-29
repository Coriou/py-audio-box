"""Source-only observation bracket tests; no Docker or installed runtime."""
import copy
import unittest

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

    def observe(self, second, later=None):
        # Real Helper.observe -> pair -> inspect -> daemon_volumes.inspect_container.
        helper=object.__new__(dh.Helper)
        helper.config=self.config
        helper.b=self.binding
        helper.family=self.family
        helper.evidence=dv.EVIDENCE
        helper.export=helper.pocket=False
        helper.read=lambda _:dict(identities={})
        counts=dict(control=0,worker=0)
        def rpc(action, request):
            if action=='facts':return dict(roots={})
            self.assertEqual(action,'inspect')
            role=request['role']
            count=counts[role]
            counts[role]+=1
            return copy.deepcopy((self.first if count==0 else second)[role])
        helper.rpc=rpc
        startups=iter((self.startup,self.startup if later is None else later))
        helper.startup=lambda:copy.deepcopy(next(startups))
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
