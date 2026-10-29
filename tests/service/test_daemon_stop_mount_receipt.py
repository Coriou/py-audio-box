"""Offline stop/restart receipt checks using real validators and inert fixtures."""
import copy
import unittest
from unittest.mock import patch

from service import daemon_helper as dh, installed_export as ex, installed_qwen_daemon as qd
from tests.service.daemon_volume_fixtures import binding, container
from tests.service.test_installed_qwen import InstalledQwenTests
from tests.service.test_installed_export import InstalledExportContractTests


def stopped(pair):
    result=copy.deepcopy(pair)
    for c in result.values():
        c['State'].update(Running=False,Status='exited',ExitCode=0,Pid=0,
                          StartedAt='2026-10-25T00:00:01Z')
    return result


def permuted(pair):
    result=copy.deepcopy(pair)
    for c in result.values():
        c['Mounts'].reverse()
        c['HostConfig']['Mounts'].reverse()
    return result


class StopMountReceiptTests(unittest.TestCase):
    def test_qwen_restart_receipt_accepts_permutation_only(self):
        fixture=InstalledQwenTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        config=fixture.config
        launch=binding(config)
        launch['policy']=qd.LAUNCH
        pair=stopped({role:container(role,config,launch,family=qd.FAMILY)
                      for role in ('control','worker')})
        history={'clean':'same retained history'}
        receipt=dict(containers=copy.deepcopy(pair),history=copy.deepcopy(history))
        current=permuted(pair)
        self.assertNotEqual(receipt['containers'],current)  # Old raw gate refused this.
        self.assertTrue(dh.matching_stopped_receipt(receipt,current,history,config,launch,qd.FAMILY))
        for change in ('source','target','rw','missing','extra','duplicate','requested_source',
                       'requested_target','requested_rw','requested_missing','requested_duplicate',
                       'image','id','incarnation','lifecycle','history','other'):
            with self.subTest(change=change):
                bad=copy.deepcopy(receipt)
                c=bad['containers']['worker']
                if change=='source':c['Mounts'][0]['Source']='/wrong'
                elif change=='target':c['Mounts'][0]['Destination']='/wrong'
                elif change=='rw':c['Mounts'][0]['RW']=not c['Mounts'][0]['RW']
                elif change=='missing':c['Mounts'].pop()
                elif change=='extra':c['Mounts'].append(dict(c['Mounts'][0],Destination='/extra'))
                elif change=='duplicate':c['Mounts'].append(copy.deepcopy(c['Mounts'][0]))
                elif change=='requested_source':c['HostConfig']['Mounts'][0]['Source']='wrong'
                elif change=='requested_target':c['HostConfig']['Mounts'][0]['Target']='/wrong'
                elif change=='requested_rw':c['HostConfig']['Mounts'][0]['ReadOnly']=not c['HostConfig']['Mounts'][0]['ReadOnly']
                elif change=='requested_missing':c['HostConfig']['Mounts'].pop()
                elif change=='requested_duplicate':c['HostConfig']['Mounts'].append(copy.deepcopy(c['HostConfig']['Mounts'][0]))
                elif change=='image':c['Image']='sha256:'+'f'*64
                elif change=='id':c['Id']='f'*64
                elif change=='incarnation':c['State']['StartedAt']='2026-10-29T00:00:00Z'
                elif change=='lifecycle':c['State']['Running']=True
                elif change=='history':bad['history']['clean']='changed'
                else:c['NetworkSettings']={'changed':True}
                self.assertFalse(dh.matching_stopped_receipt(bad,current,history,config,launch,qd.FAMILY))

    def test_export_retained_stop_validator_accepts_permutation_only(self):
        fixture=InstalledExportContractTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        config=fixture.config
        launch=binding(config)
        launch['policy']=ex.LAUNCH
        public=dict(configuration=config,launch=launch,deployment_id='a'*32,host={})
        from service import recovery
        pair=stopped({role:container(role,config,launch,family=ex.FAMILY)
                      for role in ('control','worker')})
        owners=[dict(id=c['Id'],image=c['Image'],started=c['State']['StartedAt'],role=role,
                     volumes=['control','ipc','staging'] if role=='control' else ['ipc','receipts','staging'])
                for role,c in pair.items()]
        state=dict(registrations=[dict(containers=owners)],evidence_projection=dict(data=dict(
            p1_installation=[[public['deployment_id'],1,ex.SCOPE,ex._encoded(public).decode(),ex._digest(public)]])))
        history=dict(state=state,journal={})
        receipt=dict(containers=copy.deepcopy(pair),history=copy.deepcopy(history))
        current=permuted(pair)
        self.assertNotEqual(ex._encoded(receipt),ex._encoded(dict(containers=current,history=history)))
        with patch.object(ex,'installation'),patch.object(recovery,'installed_clean_history'),patch.object(
                recovery,'installed_history_current'):
            self.assertIs(ex.stopped_receipt(receipt,current,history,public=public,family=ex.FAMILY),receipt)
            for change in ('mount','duplicate','id','image','incarnation','lifecycle','history'):
                with self.subTest(change=change):
                    bad=copy.deepcopy(receipt)
                    c=bad['containers']['worker']
                    if change=='mount':c['Mounts'][0]['Source']='/wrong'
                    elif change=='duplicate':c['Mounts'].append(copy.deepcopy(c['Mounts'][0]))
                    elif change=='id':c['Id']='f'*64
                    elif change=='image':c['Image']='sha256:'+'f'*64
                    elif change=='incarnation':c['State']['StartedAt']='2026-10-29T00:00:00Z'
                    elif change=='lifecycle':c['State']['Running']=True
                    else:bad['history']['state']['source_digest']='f'*64
                    with self.assertRaises(ValueError):
                        ex.stopped_receipt(bad,current,history,public=public,family=ex.FAMILY)


if __name__=='__main__':unittest.main()
