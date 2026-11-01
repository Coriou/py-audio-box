"""R3 detached-wire output/reuse regressions using true decoded media."""
import copy
import hashlib
import io
import pytest
import soundfile as sf
from contracts import validate as c
from contracts.media import synthetic_media
from contracts.scenarios import transcript, qwen_settings
from test_contracts import load
from test_frozen_boundary import scenario, wire, check


def rehash(t):
    t['artifact']['manifest_sha256']=c.canonical_hash(t['verified_manifest'])


def replace_media(t, frames=4800, rate=48000, channels=1, format='WAV', subtype='PCM_16'):
    data,metadata=synthetic_media(frames,rate,channels,format,subtype)
    info=sf.info(io.BytesIO(data))
    assert (info.frames,info.samplerate,info.channels,info.subtype)==(frames,rate,channels,subtype)
    assert hashlib.sha256(data).hexdigest()==metadata['sha256']
    output=t['verified_manifest']['outputs'][0];output.update(metadata)
    t['present_hashes']={output['id']:metadata['sha256']};rehash(t)
    return data


@pytest.mark.parametrize('op',['voice_prepare','voice_design'])
@pytest.mark.parametrize('seed,language,temperature',[(37,'English',0.4),(901,'French',1.2)])
def test_nondefault_exact_save_version_speech_publication(op,seed,language,temperature):
    settings=qwen_settings('qwen_clone');settings.update(seed=seed,language=language,temperature=temperature,top_k=17)
    t=transcript(op,load('production-30.json'),load('capabilities.json')['profiles'],load('operation-requests.json'),reuse_settings=settings)
    check(t)
    # All catalogue exchanges, including save and exact version-get, validate.
    import json
    ops={x['id']:x for x in json.loads((c.ROOT/'v1/operations.json').read_text())['operations']}
    for e in t['exchanges']:
        c.validate(ops[e['operation']]['request'],e['request']);c.validate(ops[e['operation']]['response'],e['response'])
    version=wire(t['exchanges'][-1]['response']);before=wire(version)
    p=load('production-30.json');p['assets'].append(wire(t['verified_manifest']['outputs'][0]))
    reuse=transcript('speech_generate',p,t['profiles'],load('operation-requests.json'),version=version)
    check(reuse)
    assert version==before==reuse['exchanges'][0]['response']==reuse['context']['plan']['voice_versions'][0]
    assert reuse['context']['plan']['resolved']['settings']==settings
    assert version['settings_sha256']==c.canonical_hash(settings)


@pytest.mark.parametrize('op',['voice_prepare','voice_design'])
@pytest.mark.parametrize('mutation',['arbitrary_digest','preimage','rehashed_preimage','configuration'])
def test_publication_reuse_preimage_bound_to_approved_plan(op,mutation):
    t=scenario(op);check(t);bad=wire(t);v=bad['verified_manifest']['candidates'][0]['version']
    if mutation=='arbitrary_digest':v['settings_sha256']='f'*64
    elif mutation=='configuration':
        candidate=bad['verified_manifest']['candidates'][0];output=bad['verified_manifest']['outputs'][0]
        old=output['id'];output['id']='other-output';candidate['audition_asset_id']=output['id']
        v['configuration']['reference_asset_id']=output['id'];v['reference_asset_ids']=[output['id']]
        bad['message']['result_asset_ids']=[output['id']];bad['artifact']['asset_ids']=[output['id']]
        bad['present_hashes']={output['id']:bad['present_hashes'][old]}
    else:
        v['settings']['seed']=82
        if mutation=='rehashed_preimage':v['settings_sha256']=c.canonical_hash(v['settings'])
    rehash(bad)
    # Worker envelope remains valid; publication must reject semantically.
    c.worker_message(bad['message'],bad['job'],bad['authorized_assets'],transport_owner='worker1',context=bad['context'])
    with pytest.raises(c.ContractError,match='settings_mismatch|reuse_configuration_mismatch|reference_mismatch'):check(bad)


@pytest.mark.parametrize('container',['voice_version','candidate','voice_result','project'])
def test_containing_resources_verify_settings_preimage(container):
    t=scenario('voice_prepare');check(t)
    v=wire(t['verified_manifest']['candidates'][0]['version']);v['settings_sha256']='f'*64
    if container=='voice_version':value=v
    elif container=='candidate':value=wire(t['verified_manifest']['candidates'][0]);value['version']=v
    elif container=='voice_result':value=wire(next(e['response'] for e in t['exchanges'] if e['operation']=='voice_results'));value['candidates'][0]['version']=v
    else:
        value=load('production-30.json');value['voice_versions'][0]['settings_sha256']='f'*64
    # Use catalogue's actual result definition name.
    kind={'candidate':'voice_candidate','voice_result':'voice_results'}.get(container,container)
    with pytest.raises(c.ContractError,match='settings_mismatch'):c.validate(kind,value)


@pytest.mark.parametrize('op',['voice_prepare','voice_design'])
@pytest.mark.parametrize('format,subtype',[('FLAC','PCM_16'),('WAV','PCM_24'),('WAV','FLOAT')])
def test_reusable_output_truthful_format_rejected(op,format,subtype):
    t=scenario(op);replace_media(t);check(t)
    bad=wire(t);replace_media(bad,format=format,subtype=subtype)
    with pytest.raises(c.ContractError,match='output_format_mismatch'):check(bad)
    if format=='FLAC':
        version=wire(bad['exchanges'][-1]['response']);p=load('production-30.json');p['assets'].append(wire(bad['verified_manifest']['outputs'][0]))
        with pytest.raises(c.ContractError,match='reference_format_mismatch'):
            transcript('speech_generate',p,bad['profiles'],load('operation-requests.json'),version=version)


@pytest.mark.parametrize('change',[dict(frames=4800,channels=1),dict(frames=146399),dict(rate=44100),dict(channels=1),dict(format='FLAC'),dict(subtype='PCM_16')])
def test_render_truthful_output_must_match_frozen_policy(change):
    t=scenario('project_render');replace_media(t,frames=146400,channels=2,subtype='PCM_24');check(t)
    bad=wire(t);args=dict(frames=146400,channels=2,subtype='PCM_24');args.update(change);replace_media(bad,**args)
    with pytest.raises(c.ContractError,match='output_policy_mismatch'):check(bad)


def test_speech_builder_never_repairs_an_invalid_version_digest():
    version=load('production-30.json')['voice_versions'][0]
    version['settings_sha256']='f'*64;before=wire(version)
    with pytest.raises(c.ContractError,match='settings_mismatch'):scenario(version=version)
    assert version==before
