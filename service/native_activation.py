"""Private native activation; completed histories require exact owner-only proofs."""
import json
import sqlite3
import threading
from pathlib import Path

from worker.transport import Refused, canonical
from service import native_package as n


class NativeActivation:
    def __init__(self, record_file, *, configuration_file, templates_file, owner_root, connector=None):
        self.record_file = record_file
        self.inputs = dict(configuration_file=configuration_file, templates_file=templates_file,
                           owner_root=owner_root, connector=connector)
        self.started = False

    def _empty_transport(self, runtime):
        # Even completed imports need the deferred native history policy.
        if (not runtime.admission_open or any(runtime.spool.iterdir()) or
                any(runtime.staging.iterdir())):
            raise Refused('native_retained_work_unsupported')

    def precheck(self, dispatcher):
        if self.started: raise Refused('native_activation_reuse')
        self.started = True
        from service.entrypoint import _INSTALLED_LIFECYCLES
        self.responsibility = dict(owner=dispatcher, closed=False, error=None)
        _INSTALLED_LIFECYCLES[id(self)] = self.responsibility
        root = n.directory(str(dispatcher.root.absolute()))
        if root != Path(self.inputs['owner_root']): raise Refused('native_owner_binding')
        self.was_empty = not any(root.iterdir())
        if not self.was_empty and not (root/'core.sqlite').is_file():
            raise Refused('native_retained_history_unsupported')
        self.saved = canonical(n.document(self.record_file))
        values = n.native_startup_inputs(self.record_file, **self.inputs, _completed_history=True)
        runtime = values['runtime']
        if runtime.staging != dispatcher.staging.absolute(): raise Refused('native_staging_binding')
        if (root == runtime.spool or root in runtime.spool.parents or runtime.spool in root.parents or
                root == runtime.staging or root in runtime.staging.parents or runtime.staging in root.parents):
            raise Refused('native_owner_storage_overlap')
        if self.was_empty: self._empty_transport(runtime)
        dispatcher.runtime = runtime
        dispatcher.production_contexts = values['production_contexts']
        dispatcher.current_profiles = values['profiles']
        dispatcher.budget = values['budget']
        if dispatcher.persistent_registration is not None: raise Refused('native_linux_registration')
        self.runtime = runtime

    def check_hello(self, hello):
        if (any(hello.get(k) != v for k,v in self.runtime.native.items()) or
                hello.get('quarantined') is not False or
                type(hello.get('launches')) is not int or hello['launches'] != self.expected_launches):
            raise Refused('native_activation_hello')

    def retained_check(self, core):
        # Core invokes this under flock, before any schema initialization/migration.
        if self.was_empty != core.fresh_initialized:
            raise Refused('native_retained_history_unsupported')
        self.expected_launches = 0 if core.fresh_initialized else completed_history(core, self.runtime)

    def postcheck(self, dispatcher):
        # Core's lifetime exclusion is held here; all SQLite remains on its owner.
        if self.was_empty != dispatcher.core.fresh_initialized:
            raise Refused('native_retained_history_unsupported')
        if dispatcher.runtime is not self.runtime: raise Refused('native_owner_binding')
        before, _, _ = n._enrollment_snapshot(**{k:v for k,v in self.inputs.items() if k != 'connector'})
        if canonical(before) != self.saved or canonical(n.document(self.record_file)) != self.saved:
            raise Refused('native_enrollment_changed')
        if not self.runtime.admission_open: raise Refused('native_retained_work_unsupported')
        self.check_hello(self.runtime.call('hello', {}))
        after, _, _ = n._enrollment_snapshot(**{k:v for k,v in self.inputs.items() if k != 'connector'})
        if canonical(after) != self.saved or canonical(n.document(self.record_file)) != self.saved:
            raise Refused('native_enrollment_changed')
        if not self.runtime.admission_open: raise Refused('native_retained_work_unsupported')

    def finished(self, dispatcher, cleanup_failed, error):
        # Keep a strong owner obligation on timeout/failed close; never retry it.
        from service.entrypoint import _INSTALLED_LIFECYCLES
        if not hasattr(self, 'responsibility') or self.responsibility['owner'] is not dispatcher: return
        self.responsibility['error'] = error
        if not cleanup_failed:
            self.responsibility['closed'] = True
            _INSTALLED_LIFECYCLES.pop(id(self), None)


def completed_history(core, runtime):
    """Read-only native policy under Core exclusion, before recovery or scheduling.

    Single-attempt successes and receipt-only failed/cancelled jobs are supported.
    Never-started jobs and terminal publication/import uncertainty still refuse.
    Clone references must remain owned and byte-identical in Core and transport.
    All reservation, filesystem and unknown evidence is retained on refusal.
    """
    from contracts import validate as oracle
    from service import recovery as r
    from service.reconciliation import Reconciliation
    from service.voice_store import _TABLE_DDL, _TRIGGER_DDL, _sql_identity
    from service.worker_transport import Authority
    from service.native_runtime import _read
    from worker import native_protocol as p
    if threading.get_ident() != runtime.owner: raise Refused('native_owner_required')
    def require(ok):
        if not ok: raise Refused('native_history_unclean')
    # Compare the exact existing canonical DDL, excluding Linux enrollment only.
    omitted = {'p1_installation', 'p1_registrations', 'p1_recovery_plans'}
    ddl = [s for s in r._INSTALLED_V1_TABLE_DDL if s.split('(')[0].split()[-1] not in omitted]
    with sqlite3.connect(':memory:') as model:
        for sql in (*ddl, *_TABLE_DDL, *_TRIGGER_DDL): model.execute(sql)
        expected = list(model.execute('SELECT type,name,tbl_name,sql FROM sqlite_master'))
    actual = core.db.execute('SELECT type,name,tbl_name,sql FROM sqlite_master').fetchall()
    identity = lambda rows: sorted((a,b,c,_sql_identity(d)) for a,b,c,d in rows)
    require(identity(actual) == identity(expected))
    columns = {k:v for k,v in r._evidence_columns().items() if k not in omitted and k not in r.B2_COLUMNS}
    data = {}; total = 0; rows_total = 0
    for table, fields in columns.items():
        names = fields.split()
        require([row[1] for row in core.db.execute('PRAGMA table_info('+table+')')] == names)
        sizes = '+'.join('coalesce(length(CAST("'+c+'" AS BLOB)),0)' for c in names)
        count, size = core.db.execute('SELECT count(*),coalesce(sum('+sizes+'),0) FROM '+table).fetchone()
        total += size
        rows_total += count
        require(rows_total <= 2048 and total <= 16*1024**2)
        data[table] = [list(row) for row in core.db.execute('SELECT * FROM '+table)]
    # Shared SQL constraint validator gets empty Linux-only tables in a detached
    # in-memory projection. No Linux history/receipt acceptance route is invoked.
    with sqlite3.connect(':memory:') as model:
        for sql in (*r._INSTALLED_V1_TABLE_DDL, *_TABLE_DDL): model.execute(sql)
        schema = list(model.execute('SELECT type,name,tbl_name,sql FROM sqlite_master'))
        schema.extend(('trigger',s.split()[2],'catalogue_versions',s) for s in _TRIGGER_DDL)
    full = dict(data, **{k:[] for k in omitted})
    r._installed_detached_rows(dict(schema=schema,data=full),dict(columns,**{k:r._evidence_columns()[k] for k in omitted}))
    typed = {t:[dict(zip(columns[t].split(),row)) for row in rows] for t,rows in data.items()}
    r._installed_detached_documents(typed)
    require(data['p1_format'] == [[1,'synthetic-core']] and data['resource_protocol'] == [[1,2]])
    require(data['p1_integrity'] == [[1,1,'healthy',None]] and not data['resource_faults'])
    require(data['resource_monitor'] == [[1,'healthy']])
    for row in typed['p1_inventory_scans']:
        summary = json.loads(row['summary'])
        require(row['complete'] == 1 and summary.get('blocked') is False and not summary.get('reasons'))
    for table in ('p1_inventory_objects','p1_inventory_history'):
        require(all(json.loads(row['record'])['classification'] in ('committed_referenced','owned_synthetic_transient') for row in typed[table]))
    # These are Dispatcher journal operations, not the wider native RPC namespace.
    commands = sorted(typed['dispatch_commands'], key=lambda row:row['seq'])
    require(all(row['op'] in ('prepare','start','observe','stop','collect','ack') and
                row['state'] == 'returned' for row in commands))
    jobs = {row['id']:row for row in typed['jobs']}
    dispatches = {row['job_id']:row for row in typed['dispatches']}
    reservations = {row['id']:row for row in typed['resource_reservations']}
    publications = {row['job_id']:row for row in typed['publications']}
    require(len(dispatches) == len(typed['dispatches']) and len(jobs) < 64)
    require(set(jobs) == set(dispatches) == set(reservations))
    require(set(publications) == {row['job_id'] for row in typed['publication_work']} <= set(jobs))
    require({d.name for d in runtime.spool.iterdir()} == {d['binding'] for d in dispatches.values()})
    for jid, stored in jobs.items():
        job = json.loads(stored['job']); context = json.loads(stored['context'])
        d = dispatches[jid]; a = Authority(*json.loads(d['authority']))
        require(d['authority'].encode() == a._snapshot and d['binding'] == a.binding)
        require(canonical(p.native_binding(a.context)) == canonical(runtime.native) and
                canonical(a.context) == canonical(context))
        require(stored['retry_of'] is None)
        require(stored['fingerprint'] == oracle.fingerprint(stored['principal'],
                stored['operation'], a.context['plan']['request']))
        ops = [row['op'] for row in commands if row['dispatch_id'] == d['seq']]
        # Finite clean-completion producer: prepare/start, observation/stop loop,
        # one collect, then ACK (including replay). _tick launches collect only
        # from a returned STOPPED observe and successful collect records outcome.
        # Observe values are not journalled: corroborate terminal proof below,
        # rather than inventing historical observation states or RPC retries.
        require(ops[:2] == ['prepare','start'] and ops.count('collect') == 1)
        collected = ops.index('collect')
        require(collected >= 3 and ops[collected-1] == 'observe' and
                all(op in ('observe','stop') for op in ops[2:collected]) and
                bool(ops[collected+1:]) and all(op == 'ack' for op in ops[collected+1:]))
        require(d['runtime'] is not None and d['fixture'] == 'qwen')
        spec = p.native_execution(context)
        require(job['state'] in ('succeeded','failed','cancelled') and len(job['attempts']) == 1 and job['attempts'][0]['reaped'] is True)
        require(job['stage'] == ('completed' if job['state'] == 'succeeded' else 'deadline_or_reaped'))
        frozen = a.job; finished = dict(job); finished['attempts'] = [dict(job['attempts'][0],reaped=False)]
        # Only these completion fields may differ from the dispatched Authority.
        immutable = lambda value: {k:v for k,v in value.items() if k not in ('state','stage','artifact_ids')}
        require(canonical(immutable(frozen)) == canonical(immutable(finished)))
        require(d['acked'] == 1 and d['phase'] == 'acknowledged' and canonical(json.loads(d['outcome'])) == canonical(job))
        receipt = json.loads(d['receipt']); p.terminal_receipt(receipt,authority=a,runtime_id=d['runtime'],local_import_uid=runtime.local_import_uid)
        retained = runtime._rpc('collect',{},authority=a,runtime_id=d['runtime'])
        require(canonical(retained['receipt']) == canonical(receipt))
        reservation = reservations[jid]; demand = json.loads(reservation['demand'])
        from service.admission import native_demand
        expected_demand = native_demand(a.context)
        require(set(demand) == set(expected_demand))
        dimensions = set(expected_demand)-{'configuration'}
        require(all(type(demand[k]) is int and 0 <= demand[k] <= 2**53-1 for k in dimensions))
        require(all(demand[k] > 0 for k in dimensions-{'vram_bytes'}))
        # Single attempt means no replacement disk charge; corroborate, never reserve.
        require(reservation['state'] == 'retained' and reservation['category'] == 'inference' and
                canonical(demand) == canonical(expected_demand))
        directory = runtime.spool/a.binding
        names = {'snapshot'}
        if spec.task == 'clone':
            # Historical decoding does not expire hardware, rederive a voice or
            # trust a retained allocation as enrollment. Recheck the exact owned
            # reference frozen in Authority and its detached input bytes.
            # Core has not attached its catalogue facade at this pre-schema
            # startup fence. The existing read-only owner check needs no DDL.
            from service.voice_store import VoiceStore
            VoiceStore(core)._asset(frozen['principal_id'], spec.reference_id)
            asset = core.asset(spec.reference_id)
            require(canonical(asset) == canonical(context['plan']['assets'][0]))
            raw = _read(core.root/'assets'/spec.reference_id, spec.reference_bytes)
            import hashlib
            require(len(raw) == spec.reference_bytes and
                    hashlib.sha256(raw).hexdigest() == spec.reference_sha256)
            from worker.native_supervisor import pcm
            pcm(raw, spec, True)
            require(_read(directory/'input', spec.reference_bytes) == raw)
            names.add('input')
        require(_read(directory/'snapshot',48000) == a._snapshot)
        if job['state'] != 'succeeded':
            # A negative receipt is the supervisor's closed decision after reap,
            # receiver join and conditioning removal. No output/import/publication
            # may be inferred away, even if the durable job says cancelled.
            require(receipt['timely'] is False and len(receipt['processes']) == 1)
            require(retained['proposal'] is None and retained['output'] is None)
            require(jid not in publications and job['artifact_ids'] == [])
            require({f.name for f in directory.iterdir()} == names)
            stage = runtime.staging/job['attempts'][0]['staging_id']
            require(stage.is_dir() and not any(stage.iterdir()))
            require(a.binding not in runtime.import_closed and
                    a.binding not in runtime.reopened_imports)
            continue
        require(receipt['timely'] is True and retained['proposal'] is not None)
        publication = publications[jid]; artifact = json.loads(publication['artifact']); manifest = json.loads(publication['manifest'])
        # Shared publication readers use scalar equality. Native retained history
        # promises exact immutable copies, including JSON numeric representation.
        owner = dict(job_id=frozen['id'], generation=frozen['generation'],
            input_sha256=frozen['input_sha256'], attempt_id=frozen['attempts'][0]['id'],
            owner_id=frozen['attempts'][0]['owner_id'], staging_id=frozen['attempts'][0]['staging_id'])
        require(canonical({k:manifest[k] for k in owner}) == canonical(owner))
        expected_artifact = dict(owner, id=manifest['artifact_id'],
            asset_ids=[o['id'] for o in manifest['outputs']],
            project_revision=0, manifest_sha256=oracle.canonical_hash(manifest),
            identity=a.context['plan']['resolved']['identity'])
        require(canonical(artifact) == canonical(expected_artifact))
        for work in typed['publication_work']:
            if work['job_id'] != jid: continue
            intent = json.loads(work['binding'])
            intent_artifact = intent['artifact'] if 'version' in intent else intent
            require(canonical(intent_artifact) == canonical(artifact))
            if 'version' in intent:
                require(canonical(intent['manifest']) == canonical(manifest))
        hashes = {}
        for asset in manifest['outputs']:
            raw = _read(core.root/'assets'/asset['id'],1048576)
            import hashlib
            hashes[asset['id']] = hashlib.sha256(raw).hexdigest()
        oracle.publication(frozen,frozen['attempts'][0],artifact,hashes,transport_owner=frozen['attempts'][0]['owner_id'],context=context,manifest=manifest,verified_result=json.loads(publication['message']),allowed_assets=a.allowed_assets)
        require(job['artifact_ids'] == [artifact['id']])
        require(any(row['id'] == artifact['id'] and canonical(json.loads(row['metadata'])) == canonical(artifact) for row in typed['artifacts']))
        directory = runtime.spool/a.binding
        names.update({'output','complete','import-intent','import-complete'})
        require({f.name for f in directory.iterdir()} == names)
        require(_read(directory/'snapshot',48000) == a._snapshot)
        pending = runtime.reopened_imports.get(a.binding)
        require(pending is not None and pending[1] is not None and pending[2] is True)
        require(pending[1]['runtime_id'] == d['runtime'])
        require(_read(directory/'import-intent',512) == _read(directory/'import-complete',512))
        preparing = context['plan']['operation'] == 'voice_prepare'
        name = 'audition.wav' if preparing else 'output.wav'
        stage = runtime.staging/job['attempts'][0]['staging_id']
        require(_read(directory/'output',1048576) == _read(stage/name,1048576))
        output = json.loads(_read(directory/'complete',256))
        if preparing:
            from worker.engines.qwen_prepare import selected_reference
            reference, _ = selected_reference(spec, _read(directory/'input',spec.reference_bytes))
            require(_read(stage/'reference.wav',1048576) == reference)
            require(len(manifest['outputs']) == 2 and manifest['outputs'][0]['sha256'] == spec.reference_sha256 and manifest['outputs'][1]['sha256'] == output['sha256'])
        else:
            require(len(manifest['outputs']) == 1 and manifest['outputs'][0]['sha256'] == output['sha256'])
        require(canonical(retained['output']) == canonical(output) and
                canonical(retained['proposal']) == canonical(dict(message=json.loads(publication['message']),manifest=manifest)))
    # Existing inventory reader checks unknown namespaces and actual retained media.
    # It performs no reconciliation writes when called directly.
    scan = Reconciliation.__new__(Reconciliation)
    scan.core,scan.db,scan.staging = core,core.db,runtime.staging
    scan.entry_limit,scan.byte_limit,scan.seconds = 2048,16*1024**2,1
    scan._collect()
    require(not scan.reasons and all(row['classification'] in ('committed_referenced','owned_synthetic_transient') for row in scan.rows.values()))
    require(not runtime.failed)
    # Only after every proof: discharge in-memory import closure, no ACK or writes.
    runtime.import_completed.update(runtime.import_closed)
    runtime.import_closed.clear()
    return len(jobs)
