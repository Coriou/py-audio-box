"""Summarize saved public-corpus records without rerunning inference."""
import argparse, json
from pathlib import Path

def combined(rows, field):
    items=[r["metrics"][field] for r in rows]
    edits=sum(x["edits"] for x in items); units=sum(x["reference_units"] for x in items)
    return dict(edits=edits,reference_units=units,rate=edits/units if units else None)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--results",type=Path,default=Path("/results")); args=parser.parse_args()
    summary={}
    for label in ["small","turbo"]:
        path=args.results/f"{label}.jsonl"
        if not path.exists(): continue
        records=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        runs=[r for r in records if r["event"]=="sample"]
        primary=[r for r in runs if r["variant"]=="primary" and r["sample_kind"]=="natural_read_speech"]
        external=args.results/f"{label}.external-memory.log"
        samples=[]
        if external.exists():
            lines=external.read_text().splitlines()
            for i,line in enumerate(lines):
                if line.startswith("2026-") and i+2<len(lines) and lines[i+1].isdigit() and lines[i+2].isdigit():
                    samples.append(dict(time=line,current_bytes=int(lines[i+1]),kernel_peak_bytes=int(lines[i+2])))
        external_memory=dict(samples=len(samples),current_peak_sampled_bytes=max((r["current_bytes"] for r in samples),default=None),largest_kernel_peak_observed_bytes=max((r["kernel_peak_bytes"] for r in samples),default=None),last_sample=samples[-1] if samples else None,scope="External10s samples; kernel peak may increase after final observation; includes container page cache")
        terminal_path=args.results/f"{label}.container.json"
        terminal=json.loads(terminal_path.read_text()) if terminal_path.exists() else None
        item=dict(environment=next((r for r in records if r["event"]=="environment"),None),load=next((r for r in records if r["event"]=="model_loaded"),None),memory=next((r for r in records if r["event"]=="memory_summary"),None),external_memory=external_memory,terminal=terminal,complete=any(r["event"]=="complete" for r in records),errors=[r for r in records if r["event"]=="error"],sample_runs=len(runs),primary_natural_count=len(primary),strata={})
        for language in ["all","fr","en"]:
            rows=primary if language=="all" else [r for r in primary if r["language_requested"]==language]
            duration=sum(r["duration_seconds"] for r in rows); elapsed=sum(r["decode_seconds"] for r in rows)
            item["strata"][language]=dict(samples=len(rows),duration_seconds=duration,decode_seconds=elapsed,rtf=elapsed/duration if duration else None,**{field:combined(rows,field) for field in ["raw_wer","normalized_wer","raw_cer","normalized_cer"]})
        item["controls"]=[dict(sample_id=r["sample_id"],variant=r["variant"],vad_filter=r["vad_filter"],gain=r["gain"],text=r["text"],decode_seconds=r["decode_seconds"],rtf=r["rtf"],metrics=r["metrics"]) for r in runs if r["sample_kind"]!="natural_read_speech" or r["variant"]!="primary"]
        summary[label]=item
    (args.results/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print(json.dumps({label:dict(complete=item["complete"],samples=item["primary_natural_count"],all=item["strata"]["all"],load=item["load"],memory=item["memory"]) for label,item in summary.items()},indent=2))

if __name__=="__main__": main()
