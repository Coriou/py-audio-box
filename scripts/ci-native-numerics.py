#!/usr/bin/env python3
"""Pinned 12Hz tiny random-parameter MPS gate; never loads pretrained weights."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def tiny_config():
    # Preserve 12Hz vocab/code-group structure, shrink width/depth/text vocabulary.
    return dict(vocab_size=3072, text_vocab_size=64, text_hidden_size=64,
        hidden_size=64, intermediate_size=128, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=128,
        max_position_embeddings=64, num_code_groups=16, codec_eos_token_id=2150,
        rope_scaling=dict(rope_type='default', mrope_section=[24,20,20]),
        code_predictor_config=dict(vocab_size=2048, hidden_size=64,
            intermediate_size=128, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=1, head_dim=128, num_code_groups=16,
            max_position_embeddings=64))


def save(path, value):
    raw=(json.dumps(value,sort_keys=True,indent=2)+'\n').encode()
    if len(raw)>65536: raise ValueError('numeric_report_limit')
    path.write_bytes(raw)


def child(path):
    report=dict(status='failed', device='mps', pretrained_weights=False,
        full_model_numerics_qualified=False, speech_decoder_tested=False, kernel_selection_profiled=False,
        gate_max_new_tokens=3, attention='sdpa',sampling=dict(temperature=.9,top_p=1.,top_k=50,repetition_penalty=1.05),
        commit=os.environ.get('GITHUB_SHA'), cases=[], config=tiny_config())
    try:
        from worker.native_child import install_spawn_guard
        install_spawn_guard()
        import torch
        if not torch.backends.mps.is_available(): raise ValueError('actual_MPS_required')
        pins={'qwen-tts':'0.1.1','transformers':'4.57.3','torch':'2.10.0'}
        report['versions']={key:importlib.metadata.version(key) for key in pins}
        if report['versions']!=pins: raise ValueError('numeric_locked_versions')
        import copy
        import inspect
        from transformers.generation.utils import GenerationMixin
        from transformers import LogitsProcessor
        from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSTalkerConfig
        from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSTalkerForConditionalGeneration
        report['sources']={str(Path(inspect.getfile(cls)).name):hashlib.sha256(Path(inspect.getfile(cls)).read_bytes()).hexdigest()
            for cls in (Qwen3TTSTalkerConfig,Qwen3TTSTalkerForConditionalGeneration,GenerationMixin)}
        report['sampling_cast']='existing transformers 4.57.3 _sample promotes returned logits to float32; no added cast'
        torch.set_num_threads(2);torch.manual_seed(2026)
        config=Qwen3TTSTalkerConfig(**report['config'])
        config._attn_implementation='sdpa';config.code_predictor_config._attn_implementation='sdpa'
        base=Qwen3TTSTalkerForConditionalGeneration(config).eval()
        report['parameter_count']=sum(p.numel() for p in base.parameters())
        if report['parameter_count']>5_000_000: raise ValueError('tiny_parameter_limit')
        def finite(value, role):
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f'nonfinite:{role}:dtype={value.dtype}:device={value.device}')
            if value.device.type!='mps': raise ValueError('unexpected_device:'+role)
        class Audit(LogitsProcessor):
            def __call__(self, input_ids, scores):
                if scores.device.type!='mps' or bool((torch.isnan(scores)|torch.isposinf(scores)).any().item()) or not bool(torch.isfinite(scores).any(-1).all().item()):
                    raise ValueError('invalid_pre_warper_sampling_logits')
                if scores.dtype!=torch.float32: raise ValueError('unexpected_upstream_sampling_dtype')
                return scores
        for name in ('float32','float16'):
            report['active_dtype']=name
            case=dict(dtype=name,requested_device='mps',raw_logit_checks=0,activation_checks=0,raw_logit_modules={},activation_modules={})
            report['cases'].append(case)
            dtype=getattr(torch,name)
            model=copy.deepcopy(base).to(device='mps',dtype=dtype)
            if any(p.dtype!=dtype or p.device.type!='mps' for p in model.parameters()): raise ValueError('parameter_dtype_or_device_fallback')
            handles=[]
            def hook(module,args,output):
                logits=getattr(output,'logits',None)
                if logits is not None:
                    finite(logits,'raw_'+type(module).__name__)
                    if logits.dtype!=dtype: raise ValueError('raw_logit_dtype_changed')
                    case['raw_logit_checks']+=1
                    role=type(module).__name__;case['raw_logit_modules'][role]=case['raw_logit_modules'].get(role,0)+1
                hidden=getattr(output,'last_hidden_state',None)
                if hidden is not None:
                    finite(hidden,'activation_'+type(module).__name__)
                    if hidden.dtype!=dtype: raise ValueError('activation_dtype_changed')
                    case['activation_checks']+=1
                    role=type(module).__name__;case['activation_modules'][role]=case['activation_modules'].get(role,0)+1
            for module in (model,model.model,model.code_predictor,model.code_predictor.model):
                handles.append(module.register_forward_hook(hook))
            x=torch.linspace(-1,1,256,device='mps',dtype=dtype).reshape(1,4,64)
            with torch.inference_mode():
                result=model.generate(inputs_embeds=x,attention_mask=torch.ones((1,4),device='mps',dtype=torch.long),
                    trailing_text_hidden=x,tts_pad_embed=x[:,:1],max_new_tokens=3,min_new_tokens=2,
                    do_sample=True,top_k=50,top_p=1.,temperature=.9,repetition_penalty=1.05,
                    subtalker_dosample=True,subtalker_top_k=50,subtalker_top_p=1.,subtalker_temperature=.9,
                    eos_token_id=2150,pad_token_id=2150,
                    suppress_tokens=[i for i in range(2048,3072) if i!=2150],
                    output_hidden_states=True,output_scores=True,return_dict_in_generate=True,
                    logits_processor=[Audit()])
                torch.mps.synchronize()
                for scores in result.scores:
                    # Suppression/top-k intentionally use -inf. NaN/+inf are not valid.
                    if bool((torch.isnan(scores)|torch.isposinf(scores)).any().item()): raise ValueError('nonfinite_processed_scores')
                    probabilities=torch.softmax(scores,dim=-1)
                    finite(probabilities,'post_warper_probabilities')
                    if not bool((probabilities.sum(-1)-1).abs().lt(1e-4).all().item()): raise ValueError('invalid_probability_mass')
                tokens=result.sequences
                if tokens.device.type!='mps' or not bool(((tokens>=0)&(tokens<3072)).all().item()): raise ValueError('invalid_sampled_token')
                case.update(steps=len(result.scores),sampling_dtype=str(result.scores[0].dtype),
                    selected_device=str(next(model.parameters()).device),
                    selected_attention=[model.config._attn_implementation,model.code_predictor.config._attn_implementation],status='passed')
                if case['selected_attention']!=['sdpa','sdpa']: raise ValueError('attention_fallback')
                if case['raw_logit_checks']<3 or case['activation_checks']<3: raise ValueError('12hz_paths_not_exercised')
            for handle in handles: handle.remove()
            del model,result
        # Expected hazard, not a model failure: a post-projection cast cannot fix inf.
        a=torch.full((1,1),65504.,device='mps',dtype=torch.float16)
        overflow=a@torch.full((1,1),2.,device='mps',dtype=torch.float16)
        report['overflow_sentinel']=dict(dtype=str(overflow.dtype),device=str(overflow.device),
            value=str(overflow.item()),finite_after_float32_cast=bool(torch.isfinite(overflow.float()).all().item()),
            mathematical_value=131008,scope='expected representational hazard, not trained-model evidence')
        try: finite(torch.full((1,1),float('nan'),device='mps'), 'injected_nan_sentinel')
        except ValueError: report['nonfinite_sentinel_rejected']=True
        else: raise ValueError('nonfinite_checker_failed')
        report['status']='passed'
    except BaseException as exc:
        report['failure']=dict(type=type(exc).__name__,message=str(exc)[:2000],dtype=report.get('active_dtype'),device='mps')
        raise
    finally:
        save(path,report)


def validate_report(report):
    if (report.get('status')!='passed' or report.get('full_model_numerics_qualified') is not False
            or report.get('pretrained_weights') is not False or report.get('speech_decoder_tested') is not False
            or report.get('nonfinite_sentinel_rejected') is not True):
        raise ValueError('numeric_gate_report_scope')
    if [case.get('dtype') for case in report['cases']] != ['float32','float16']:
        raise ValueError('numeric_gate_dtype_cases')
    for case in report['cases']:
        if (case.get('status')!='passed' or not case.get('selected_device','').startswith('mps')
                or case.get('sampling_dtype')!='torch.float32' or not 1<=case.get('steps',0)<=3
                or case.get('raw_logit_modules',{}).get('Qwen3TTSTalkerForConditionalGeneration',0)<2
                or case.get('raw_logit_modules',{}).get('Qwen3TTSTalkerCodePredictorModelForConditionalGeneration',0)<15
                or case.get('activation_checks',0)<3):
            raise ValueError('numeric_gate_paths_or_device')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--child',action='store_true')
    args=parser.parse_args()
    if args.child: return child(args.output)
    args.output.mkdir(parents=True,exist_ok=False)
    spec=importlib.util.spec_from_file_location('numeric_owner',ROOT/'scripts/ci-apple-speech.py')
    owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
    observation={}
    env={k:v for k,v in os.environ.items() if k in ('PATH','HOME','TMPDIR','GITHUB_SHA')}
    env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTORCH_ENABLE_MPS_FALLBACK='0',TOKENIZERS_PARALLELISM='false')
    try:
        owner.owned_child([sys.executable,'-I','-B','-u',str(Path(__file__).resolve()),'--child','--output',str(args.output/'report.json')],env,args.output/'child.log',observation,seconds=90)
        validate_report(json.loads((args.output/'report.json').read_text()))
    except BaseException:
        for name in ('report.json','child.log'):
            file=args.output/name
            if file.exists():
                print(name+': '+file.read_bytes()[:16384].decode('utf-8','replace'),file=sys.stderr)
        raise
    finally: save(args.output/'owner.json',observation)


if __name__=='__main__': main()
