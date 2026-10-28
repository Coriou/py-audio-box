"""CPU-only prospective metadata/spool accounting; no Qwen epoch authority."""
from worker import native_cpu_protocol as p
from worker.transport import Refused


def preallocate(core,context):
    spec=p.native_execution(context)
    from worker.native_cpu_qualification import current,MAX_JOBS
    session=current()
    limit=MAX_JOBS if session is not None else 63
    if core.db.execute('SELECT count(*) FROM jobs').fetchone()[0]>=limit:raise Refused('native_cpu_history_capacity')
    runtime=getattr(core,'_native_retention_runtime',None)
    if runtime is None:raise Refused('native_cpu_owner_required')
    runtime._account(3*spec.max_output_bytes+96000,new=True)
    from service import recovery
    rows=size=0
    for table,fields in recovery._evidence_columns().items():
        if core.db.execute('SELECT 1 FROM sqlite_master WHERE type="table" AND name=?',(table,)).fetchone() is None:continue
        charge='+'.join('coalesce(length(CAST("'+c+'" AS BLOB)),0)' for c in fields.split())
        count,bytes_=core.db.execute('SELECT count(*),coalesce(sum('+charge+'),0) FROM '+table).fetchone()
        rows+=count;size+=bytes_
    if rows+64>2048 or size+8*48000>16*1024**2:raise Refused('native_cpu_history_capacity')
