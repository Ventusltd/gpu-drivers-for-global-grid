"""Local unsupervised lexical model; proposals only, never executes target source."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
import argparse, collections, datetime as dt, hashlib, json, math, posixpath, re, subprocess, time
from pathlib import Path
import numpy as np

def git(root,*args): return subprocess.check_output(['git','-C',str(root),*args],timeout=120)
def save(path,value):
    raw=json.dumps(value,indent=2).encode()
    if len(raw)>8_000_000: raise ValueError('JSON partition exceeds8MB')
    path.write_bytes(raw)
def words(text):
    # Remove quoted literals before feature extraction; this is lexical preprocessing,
    # not a language parser or a proof of secret removal. Models stay offline.
    text=re.sub(r'''"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`''',' ',text)
    return [w.lower() for w in re.findall(r'[A-Za-z_][A-Za-z_0-9]{2,79}',text) if w.lower() not in {'const','function','return','import','from','export','default','true','false','null','none','self','this','else','async','await','undefined','var','let','for','while','class','def','try','catch','with','new'}]
def train(counts):
    df=collections.Counter(t for row in counts for t in row)
    vocabulary=[t for t,n in df.most_common(2048) if n<max(3,len(counts)*.95)] or list(df)[:2048]
    lookup={t:i for i,t in enumerate(vocabulary)}
    x=np.zeros((len(counts),len(vocabulary)),dtype=np.float32)
    idf=np.array([math.log((1+len(counts))/(1+df[t]))+1 for t in vocabulary],dtype=np.float32)
    for i,row in enumerate(counts):
        for term,n in row.items():
            if term in lookup:x[i,lookup[term]]=1+math.log(n)
    x*=idf;x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
    return x,vocabulary,idf
def cluster(x,seed):
    if len(x)==0 or x.shape[1]==0:return np.zeros(len(x),dtype=int),np.empty((0,x.shape[1])),0
    k=min(12,len(x),max(1,int(math.sqrt(len(x)/2))))
    rng=np.random.default_rng(seed);centres=x[rng.choice(len(x),k,replace=False)].copy();labels=np.zeros(len(x),dtype=int)
    for iteration in range(30):
        new=np.argmax(x@centres.T,axis=1)
        if iteration and np.array_equal(new,labels):break
        labels=new
        for j in range(k):
            subset=x[labels==j]
            if len(subset):centres[j]=subset.mean(axis=0)
        centres/=np.maximum(np.linalg.norm(centres,axis=1,keepdims=True),1e-12)
    return labels,centres,iteration+1
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--reuse',type=Path);p.add_argument('--commit');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    def phase(name):save(a.out/'phase.json',{'phase':name,'at':dt.datetime.now(dt.timezone.utc).isoformat()})
    if a.reuse:
        phase('Load pinned learned model');meta=json.loads((a.reuse/'manifest.json').read_text());files=json.loads((a.reuse/'files.json').read_text());vocabulary=json.loads((a.reuse/'vocabulary.json').read_text());x=np.concatenate([np.load(a.reuse/f,allow_pickle=False)['x'] for f in meta['vectorShards']],axis=0)
    else:
        phase('Pin inventory');head=git(a.root,'rev-parse',a.commit or 'HEAD').decode().strip();entries=[]
        for row in git(a.root,'ls-tree','-rlz',head).split(b'\0'):
            if not row:continue
            info,name=row.split(b'\t',1);fields=info.decode().split();name=name.decode('utf-8',errors='replace')
            if len(fields)==4 and fields[1]=='blob':entries.append((fields[2],int(fields[3]),name))
        tracked={e[2] for e in entries};seen=set();eligible=[];excluded=collections.Counter()
        for sha,size,name in entries:
            reason=None
            if Path(name).suffix.lower() not in {'.js','.mjs','.cjs','.py','.ts','.tsx','.html','.css','.yml','.yaml'}:reason='not source type'
            elif any(part in {'node_modules','vendor','data','results','versions','receipts','homepage_versions'} for part in name.lower().split('/')) or re.search(r'(?:^|/)\d{12}(?:/|$)',name):reason='archived/vendor/data scope'
            elif size>2_000_000:reason='source blob over2MB'
            elif sha in seen:reason='exact duplicate blob'
            if reason:excluded[reason]+=1;continue
            seen.add(sha);eligible.append((sha,size,name))
        selected=eligible[:5000];phase('Learn lexical features and trace relative imports');files=[];counts=[];unresolved=[]
        for sha,size,name in selected:
            raw=git(a.root,'cat-file','blob',sha);text=raw.decode('utf-8',errors='replace');count=collections.Counter(words(text));counts.append(dict(count.most_common(500)))
            imports=sorted(set(re.findall(r'''(?:from\s*|require\(\s*|import\(\s*)['"]([^'"]+)''',text)))
            local=[]
            for ref in imports:
                if not ref.startswith('.'):continue
                target=posixpath.normpath(posixpath.join(posixpath.dirname(name),ref.split('?')[0]));options=[target]+[target+s for s in ['.js','.mjs','.ts','/index.js','/index.mjs']];resolved=next((q for q in options if q in tracked),None);local.append({'import':ref,'resolved':resolved})
                if resolved is None:unresolved.append({'path':name,'import':ref,'status':'static unresolved; bundler/runtime mapping may explain'})
            files.append({'path':name,'blob':sha,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'lines':len(text.splitlines()),'relativeImports':local,'termsBeforeCap':len(count),'termsRetained':min(500,len(count))})
        x,vocabulary,idf=train(counts);del counts
        phase('Five-month commit history');churn=collections.Counter();commit_count=0
        proc=subprocess.Popen(['git','-C',str(a.root),'log',head,'--since=153 days ago','--format=COMMIT:%H','--numstat'],stdout=subprocess.PIPE,text=True,encoding='utf-8',errors='replace')
        for line in proc.stdout:
            if line.startswith('COMMIT:'):commit_count+=1
            else:
                parts=line.rstrip('\n').split('\t',2)
                if len(parts)==3:churn[parts[2]]+=1
        if proc.wait():raise RuntimeError('History scan failed')
        shards=[]
        for offset in range(0,len(x),512):
            name=f'vectors-{offset:05d}.npz';np.savez_compressed(a.out/name,x=x[offset:offset+512]);shards.append(name)
        # Empty owners still get a model shard so subsequent load remains explicit.
        if not shards:np.savez_compressed(a.out/'vectors-00000.npz',x=x);shards=['vectors-00000.npz']
        np.savez_compressed(a.out/'idf.npz',idf=idf)
        meta={'schema':'ventus.lexical-learning.v1','repository':a.root.name,'commit':head,'trackedFiles':len(entries),'eligible':len(eligible),'selected':len(selected),'unselected':max(0,len(eligible)-len(selected)),'exclusions':dict(excluded),'dimensions':len(vocabulary),'vectorShards':shards,'historyCommits':commit_count,'historyDays':153,'hotspots':[{'path':p,'touches':n} for p,n in churn.most_common(30)],'dirtyPaths':len(git(a.root,'status','--porcelain').splitlines()),'scope':'Local committed source only. TF-IDF and spherical k-means learn lexical groupings, not semantics, correctness or proven duplication. No target-owned code executed.'}
        save(a.out/'manifest.json',meta);save(a.out/'files.json',files);save(a.out/'vocabulary.json',vocabulary);save(a.out/'unresolved-imports.json',unresolved)
    phase('Cluster and measure assignment stability');labels,centres,iterations=cluster(x,a.seed);groups=[]
    for j,c in enumerate(centres):
        ids=np.where(labels==j)[0];representatives=sorted(ids,key=lambda i:float(x[i]@c),reverse=True)[:8]
        groups.append({'cluster':j,'files':len(ids),'terms':[vocabulary[i] for i in np.argsort(c)[-8:][::-1]],'representatives':[files[i]['path'] for i in representatives]})
    # Compare neighbour co-membership rather than raw cluster IDs, which can permute.
    stability=None
    if a.reuse and (a.reuse/'labels.npz').exists():
        baseline=np.load(a.reuse/'labels.npz',allow_pickle=False)['labels'];rng=np.random.default_rng(42)
        if len(labels)>1:
            pairs=rng.integers(0,len(labels),size=(min(10000,len(labels)**2),2));valid=pairs[:,0]!=pairs[:,1];pairs=pairs[valid];same=baseline[pairs[:,0]]==baseline[pairs[:,1]];current=labels[pairs[:,0]]==labels[pairs[:,1]]
            stability={'samplePairs':len(pairs),'coMembershipAgreement':float(np.mean(same==current)),'baselineSameClusterPairs':int(same.sum()),'warning':'Agreement is dominated by different-cluster pairs; not semantic accuracy.'}
    np.savez_compressed(a.out/'labels.npz',labels=labels)
    result={'repository':meta['repository'],'commit':meta['commit'],'seed':a.seed,'iterations':iterations,'groups':groups,'stability':stability,'seconds':round(time.monotonic()-start,2),'status':'review candidates only'};save(a.out/'clusters.json',result)
    lines=[f'# {meta["repository"]}: lexical review card',f'Commit {meta["commit"]}; seed{a.seed}; {len(files)} source blobs. Not a correctness verdict.','']
    for g in groups:lines+=['- '+', '.join(g['terms'])+': '+str(g['files'])+' files; start at '+', '.join(g['representatives'][:3])]
    (a.out/'REVIEW.md').write_text('\n'.join(lines),encoding='utf-8');phase('Complete');print(json.dumps({'repository':meta['repository'],'files':len(files),'groups':len(groups),'seconds':result['seconds']}))
if __name__=='__main__':main()
