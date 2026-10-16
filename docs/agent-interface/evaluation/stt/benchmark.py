"""Bounded serial CPU ASR probe; task-owned public inputs only."""
import argparse, hashlib, importlib.metadata, json, os, platform, re, resource, threading, time, unicodedata
from pathlib import Path

def errors(a, b):
    prev = list(range(len(b)+1))
    for i,x in enumerate(a,1):
        curr=[i]
        for j,y in enumerate(b,1): curr.append(min(curr[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
        prev=curr
    return dict(edits=prev[-1], reference_units=len(a), rate=prev[-1]/len(a) if a else None, inserted_when_empty=len(b) if not a else None)

def normalize(s): return " ".join(re.findall(r"\w+", unicodedata.normalize("NFC",s).casefold()))

def metrics(ref, hyp):
    a,b=normalize(ref),normalize(hyp)
    return dict(raw_wer=errors(ref.split(),hyp.split()), normalized_wer=errors(a.split(),b.split()), raw_cer=errors(ref,hyp), normalized_cer=errors(a,b), normalization="nfc-casefold-word-tokens-v1;digits-and-accents-preserved")

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--model",choices=["small","turbo"],required=True); parser.add_argument("--assets",type=Path,default=Path("/assets")); parser.add_argument("--out",type=Path,default=Path("/results")); parser.add_argument("--threads",type=int,default=4)
    args=parser.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((args.assets/"manifest.json").read_text())
    target=args.out/f"{args.model}.jsonl"
    started=time.perf_counter(); cgroup=[]; done=threading.Event()
    def memory():
        while not done.wait(.2):
            try: cgroup.append(int(Path("/sys/fs/cgroup/memory.current").read_text()))
            except OSError: pass
    threading.Thread(target=memory,daemon=True).start()
    def record(payload):
        payload.update(model=args.model,wall_since_start=time.perf_counter()-started)
        with target.open("a") as f: f.write(json.dumps(payload,ensure_ascii=False)+"\n"); f.flush(); os.fsync(f.fileno())
        print(json.dumps({k:v for k,v in payload.items() if k not in ("text","reference","metrics")}),flush=True)
    try:
        from faster_whisper import WhisperModel
        import ctranslate2
        import numpy as np
        import soundfile as sf
        versions={p:importlib.metadata.version(p) for p in ["faster-whisper","ctranslate2","av","onnxruntime","numpy"]}
        record(dict(event="environment",versions=versions,platform=platform.platform(),supported_cpu_compute_types=sorted(ctranslate2.get_supported_compute_types("cpu")),threads=args.threads,decoder_beam=5,batch_size=1,max_new_tokens=128,model_identity=manifest["models"][args.model]))
        t=time.perf_counter()
        model=WhisperModel(str(args.assets/"models"/args.model),device="cpu",compute_type="int8",cpu_threads=args.threads,num_workers=1,local_files_only=True)
        record(dict(event="model_loaded",load_seconds=time.perf_counter()-t,load_count=1,requested_compute_type="int8",effective_compute_type=str(getattr(model.model,"compute_type","not_exposed")),device=str(getattr(model.model,"device","not_exposed"))))
        natural=[s for s in manifest["samples"] if s["kind"]=="natural_read_speech"]
        rest=[s for s in manifest["samples"] if s["kind"]!="natural_read_speech"]
        plan=[(s,False,"primary",1) for s in natural+rest]
        plan += [(natural[0],False,"warm_repeat",1)]
        plan += [(next(s for s in natural if s["language"]==lang),True,"vad_ablation",1) for lang in ["fr","en"]]
        plan += [(s,True,"vad_ablation",1) for s in rest if s["kind"]=="synthetic_negative"]
        plan += [(natural[0],v,"low_amplitude_ablation",.02) for v in [False,True]]
        for index,(sample,vad,variant,gain) in enumerate(plan):
            t=time.perf_counter(); signal,rate=sf.read(sample["file"],dtype="float32")
            if rate!=16000 or signal.ndim!=1: raise ValueError("Evaluation corpus must be mono16k")
            signal=signal*gain
            segments,info=model.transcribe(signal,language=sample["language"],task="transcribe",beam_size=5,vad_filter=vad,word_timestamps=False,condition_on_previous_text=True,temperature=0,max_new_tokens=128)
            segments=list(segments); text="".join(s.text for s in segments).strip(); elapsed=time.perf_counter()-t
            record(dict(event="sample",index=index,sample_id=sample["id"],sample_kind=sample["kind"],variant=variant,gain=gain,cache_state="cold_first_decode" if index==0 else "warm_model",language_requested=sample["language"],language_detected=info.language,language_probability=info.language_probability,vad_filter=vad,duration_seconds=sample["duration"],duration_after_vad=info.duration_after_vad,decode_seconds=elapsed,rtf=elapsed/sample["duration"],text=text,reference=sample["text"],metrics=metrics(sample["text"],text),segments=[dict(start=s.start,end=s.end,avg_logprob=s.avg_logprob,no_speech_prob=s.no_speech_prob) for s in segments],maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        record(dict(event="complete",sample_runs=len(plan),total_seconds=time.perf_counter()-started))
    except BaseException as exc:
        record(dict(event="error",error_type=type(exc).__name__,message=str(exc)))
        raise
    finally:
        done.set()
        try: kernel_peak=int(Path("/sys/fs/cgroup/memory.peak").read_text())
        except OSError: kernel_peak=None
        record(dict(event="memory_summary",process_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,cgroup_memory_current_peak_sampled_bytes=max(cgroup,default=None),cgroup_memory_peak_kernel_bytes=kernel_peak,cgroup_sampling_seconds=.2,cgroup_samples=len(cgroup),scope="whole dedicated container cgroup including page cache; sampled peak may miss spikes; RSS not container memory"))

if __name__=="__main__": main()
