"""Bounded local learning queue through a fixed deadline; no source mutations/deployments."""
import argparse,datetime as dt,json,os,shutil,subprocess,sys,time
from pathlib import Path
import psutil
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--until',required=True);p.add_argument('--roots',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
deadline=dt.datetime.fromisoformat(a.until).timestamp()
if not time.time()<deadline<=time.time()+12*3600:raise ValueError('Deadline must be within12hours')
here=Path(__file__).parent;owners=json.loads(a.roots.read_text());roots=[Path(r) for r in owners if (Path(r)/'.git').exists()]
start=time.time();active={};jobs=[];queued=set();hour=-1;completed=0;failed=0
def save(name,value):
    target=a.out/name;tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2),encoding='utf-8');os.replace(tmp,target)
def phase(path):
    try:return json.loads(path.read_text()).get('phase')
    except (OSError,ValueError):return 'starting or writing checkpoint'
def end(proc):
    try:
        root=psutil.Process(proc.pid);children=root.children(recursive=True)
        for child in children:child.terminate()
        root.terminate();_,alive=psutil.wait_procs(children+[root],timeout=3)
        for child in alive:child.kill()
    except psutil.Error:pass
save('PLAN.json',{'deadlineUTC':a.until,'repositories':[str(r) for r in roots],'initialStages':['commit inventory','lexical TF-IDF learning','relative dependency scan','five-month history','spherical clustering','review card'],'initialStageCount':6*len(roots),'hourly':'Repeat clustering with independent seeds on pinned models; detect changed local HEADs and reindex. No fabricated work or automatic deployment.','storageLimitBytes':350_000_000,'scope':'Unsupervised lexical learning, not training an LLM, semantic proof or code correctness.'})
try:
    while time.time()<deadline and not (a.out/'STOP').exists():
        current=int((time.time()-start)//3600)
        if current!=hour:
            hour=current
            for root in roots:
                head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True,timeout=20).strip()
                base=a.out/'models'/root.name/head
                mode='stability' if (base/'manifest.json').exists() else 'learn'
                key=(str(root),head,hour)
                if key not in queued:jobs.append({'root':str(root),'head':head,'base':str(base),'hour':hour,'mode':mode});queued.add(key)
        ram=psutil.virtual_memory().available/1024**3;battery=psutil.sensors_battery();reason=None
        if ram<2.5:reason='RAM reserve'
        if battery and not battery.power_plugged:reason='AC power required'
        if shutil.disk_usage(a.out).free<5*1024**3:reason='Disk reserve'
        if sum(f.stat().st_size for f in a.out.rglob('*') if f.is_file())>350_000_000:reason='350MB evidence budget';save('BUDGET-STOP.json',{'at':time.time()});break
        for pid,record in list(active.items()):
            proc=record['process'];job=record['job'];expired=time.time()-record['started']>900
            if reason or expired:end(proc)
            code=proc.poll()
            if code is not None:
                record['log'].close();ok=code==0 and (record['out']/'clusters.json').exists();completed+=int(ok);failed+=int(not ok)
                receipt={**job,'exitCode':code,'ok':ok,'reason':reason or ('15minute job timeout' if expired else None),'output':str(record['out'])}
                save(f'job-{len(queued):04d}-{pid}.json',receipt);del active[pid]
        if not reason:
            while jobs and len(active)<2 and psutil.virtual_memory().available/1024**3>3.2:
                job=jobs.pop(0);base=Path(job['base']);out=base if job['mode']=='learn' else a.out/'stability'/Path(job['root']).name/f'hour-{hour:02d}'
                out.mkdir(parents=True,exist_ok=True);cmd=[sys.executable,str(here/'learn-repo.py'),'--root',job['root'],'--out',str(out),'--seed',str(job['hour']),'--commit',job['head']]
                if job['mode']=='stability':cmd+=['--reuse',str(base)]
                log=(out/'worker.log').open('w',encoding='utf-8');proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW,env=dict(os.environ,OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2'))
                try:handle=psutil.Process(proc.pid);handle.cpu_affinity(list(range(min(8,psutil.cpu_count()))));handle.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                except psutil.Error:pass
                active[proc.pid]={'process':proc,'log':log,'job':job,'started':time.time(),'out':out}
        save('STATUS.json',{'at':dt.datetime.now(dt.timezone.utc).isoformat(),'pid':os.getpid(),'deadlineUTC':a.until,'hour':hour,'completedJobs':completed,'failedJobs':failed,'queuedNow':len(jobs),'active':[{'pid':pid,'repository':Path(r['job']['root']).name,'mode':r['job']['mode'],'phase':phase(r['out']/'phase.json')} for pid,r in active.items()],'availableGiB':ram,'state':reason or ('learning' if active else 'watching until next hourly robustness sweep'),'futureHourlySweeps':max(0,int((deadline-time.time())//3600))})
        time.sleep(3 if active or jobs else 15)
finally:
    for r in active.values():end(r['process']);r['log'].close()
    save('FINAL.json',{'finishedUTC':dt.datetime.now(dt.timezone.utc).isoformat(),'completedJobs':completed,'failedJobs':failed,'remaining':jobs,'scope':'Review candidates only; no deployments performed.'})
