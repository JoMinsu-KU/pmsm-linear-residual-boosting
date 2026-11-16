"""Reanalyse paired predictions from an explicitly supplied completed portable run.

No training, prediction, model deserialization, original campaign execution or
network operation occurs. Source hashes and complete endpoint pairs are checked.
"""
from pathlib import Path
import argparse, csv, hashlib, json, math, os
import numpy as np

METHODS=('linear_residual','direct200','d9')
LABELS={'linear_residual':'Linear residual boosting','direct200':'Matched direct boosting','d9':'Selected direct boosting'}
COHORTS=(('ipmsm','calibration'),('ipmsm','pilot'),('ipmsm','confirmation'),('paderborn','development'),('paderborn','confirmation'))
ROLE={'calibration':'Calibration','pilot':'Pilot evaluation','development':'Development','confirmation':'Historical evaluation'}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def write_csv(path,rows):
    with Path(path).open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def exact(a,b,label):
    if not np.array_equal(a,b):raise ValueError(label)

def profile_metrics_pair(ds,role,pid,index,target,original,own):
    """Pure paired arithmetic: arrays keyed by the three canonical method IDs."""
    y=np.asarray(target);ix=np.asarray(index);n=len(ix)
    if n<5 or y.shape!=ix.shape or np.any(np.diff(ix)!=1):raise ValueError('Contiguous paired retained endpoints required')
    segments=[];pairs=[];whole=[]
    for method in METHODS:
        po=original[method];pn=own[method]
        if po.shape!=y.shape or pn.shape!=y.shape or not np.isfinite([po,pn,y]).all():raise ValueError('Finite paired prediction vectors')
        mo=float(np.mean((po-y)**2));mn=float(np.mean((pn-y)**2))
        whole.append(dict(dataset=ds,role=role,profile=pid,method=method,n=n,
            original_rmse=math.sqrt(mo),own_rmse=math.sqrt(mn),difference=math.sqrt(mn)-math.sqrt(mo),original_mse=mo,own_mse=mn))
        for b in range(5):
            lo=n*b//5;hi=n*(b+1)//5
            common=dict(dataset=ds,role=role,profile=pid,method=method,fifth=b+1,n=hi-lo,
                        first_source_index=int(ix[lo]),last_source_index=int(ix[hi-1]))
            mse0=float(np.mean((po[lo:hi]-y[lo:hi])**2));mse1=float(np.mean((pn[lo:hi]-y[lo:hi])**2))
            for arm,mse in (('original',mse0),('own_boundary',mse1)):
                segments.append({**common,'arm':arm,'mse':mse,'rmse':math.sqrt(mse)})
            pairs.append({**common,'original_rmse':math.sqrt(mse0),'own_rmse':math.sqrt(mse1),
                          'difference':math.sqrt(mse1)-math.sqrt(mse0),'original_mse':mse0,'own_mse':mse1})
    return segments,pairs,whole

def aggregate_pairs(pairs,whole):
    """Pure profile-macro and pooled aggregation for the five complete cohorts."""
    summaries=[];whole_summary=[]
    for ds,role in COHORTS:
        for method in METHODS:
            for fifth in range(1,6):
                rows=[r for r in pairs if (r['dataset'],r['role'],r['method'],r['fifth'])==(ds,role,method,fifth)]
                if not rows:raise ValueError('Missing configured cohort or method')
                n=sum(r['n'] for r in rows)
                summaries.append(dict(dataset=ds,role=role,method=method,fifth=fifth,profiles=len(rows),n=n,
                    original_macro_rmse=math.fsum(r['original_rmse'] for r in rows)/len(rows),
                    own_macro_rmse=math.fsum(r['own_rmse'] for r in rows)/len(rows),
                    macro_difference=math.fsum(r['difference'] for r in rows)/len(rows),
                    original_pooled_rmse=math.sqrt(math.fsum(r['original_mse']*r['n'] for r in rows)/n),
                    own_pooled_rmse=math.sqrt(math.fsum(r['own_mse']*r['n'] for r in rows)/n),
                    improved_profiles=sum(r['difference']<0 for r in rows),worsened_profiles=sum(r['difference']>0 for r in rows)))
            rows=[r for r in whole if (r['dataset'],r['role'],r['method'])==(ds,role,method)]
            whole_summary.append(dict(dataset=ds,role=role,method=method,profiles=len(rows),n=sum(r['n'] for r in rows),
                original_macro_rmse=math.fsum(r['original_rmse'] for r in rows)/len(rows),
                own_macro_rmse=math.fsum(r['own_rmse'] for r in rows)/len(rows)))
    return summaries,whole_summary

def save_metrics(out,segments,pairs,whole,summaries,whole_summary):
    for name,rows in [('profiles',segments),('paired',pairs),('summary',summaries),('whole-profile',whole),('whole-summary',whole_summary)]:
        write_csv(out/f'analysis/initialization-{name}.csv',rows)



def compute_from_run(run_dir,out):
    """Read completed portable-pipeline stages, using no historical archive."""
    root=Path(run_dir).resolve();out=Path(out).resolve()
    if out.exists():raise FileExistsError('Choose a new --output directory')
    if out.is_relative_to(root):raise ValueError('Keep analysis output outside the immutable run directory')
    run=read(root/'run.json');sources={}
    binding={name:run[name] for name in ('schema','mode','datasets','numerical_runtime','device','files')}
    expected=hashlib.sha256(json.dumps(binding,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if run['key']!=expected:raise ValueError('Run binding mismatch: metadata changed')
    if run['schema']!='portable-study-run-v2':raise ValueError('Unsupported run schema')
    if run['mode'] not in ('smoke','paper'):raise ValueError('Unknown run mode')
    if not {'ipmsm','paderborn'}.issubset(run['datasets']):raise ValueError('Both study datasets are required')
    def key(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    def verified_stage(name):
        folder=root/name
        if folder.is_symlink() or not folder.resolve().is_relative_to(root):raise ValueError('Unsafe stage path')
        receipt=read(folder/'complete.json')
        if receipt['input_key']!=key({'run':run['key'],'stage':name,'extra':None}):raise ValueError('Stage belongs to another run')
        listed=[r['path'] for r in receipt['files']]
        actual={p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file() and p.name!='complete.json'}
        if len(listed)!=len(set(listed)) or set(listed)!=actual:raise ValueError('Stage file inventory mismatch')
        for item in receipt['files']:
            path=folder/item['path']
            if not path.resolve().is_relative_to(folder.resolve()) or path.is_symlink() or sha(path)!=item['sha256']:
                raise ValueError('Stage changed or unsafe path')
            sources[path.relative_to(root).as_posix()]=item['sha256']
        sources[(folder/'complete.json').relative_to(root).as_posix()]=sha(folder/'complete.json')
        return folder
    config=read(verified_stage('report')/'metadata.json')['configuration']
    if config.get('spans')!=[16,64,256,1024,4096]:raise ValueError('Unsupported filter spans')
    for ds in ('ipmsm','paderborn'):
        spec=config['datasets'][ds]
        if any(spec.get(name)!=value for name,value in [('warmup',300),('train_stride',5),('evaluation_stride',1)]):
            raise ValueError('Unsupported warmup/stride configuration: '+ds)
    segments=[];pairs=[];whole=[];model_checks=[]
    for ds in ('ipmsm','paderborn'):
        roles=config['datasets'][ds]['roles']
        cache=verified_stage(f'cache/{ds}');cachemeta=read(cache/'dataset.json')
        if cachemeta['dataset']!=ds or cachemeta['roles']!=roles:raise ValueError('Cache partition differs')
        cache_profiles={int(item['id']):item for item in cachemeta['profiles']}
        expected_ids={int(pid) for ids in roles.values() for pid in ids}
        if set(cache_profiles)!=expected_ids or len(cache_profiles)!=len(cachemeta['profiles']):raise ValueError('Incomplete or duplicate cached profile set')
        original=verified_stage(f'evaluation/{ds}/original')
        own=verified_stage(f'evaluation/{ds}/own_boundary')
        old_model=read(verified_stage(f'models/{ds}/original/cpu')/'cpu/model.json')
        own_model=read(verified_stage(f'models/{ds}/own_boundary/cpu')/'cpu/model.json')
        for arm,model in [('original',old_model),('own_boundary',own_model)]:
            if model['dataset']!=ds or model['metadata']['arm']!=arm:raise ValueError('Model dataset/arm mismatch')
            if model['metadata']['fit_ids']!=roles['fit']:raise ValueError('Model training IDs differ from run configuration')
        for method,component in [('linear_residual','residual'),('direct200','matched_direct'),('d9','selected_direct')]:
            if old_model['metadata']['configurations'][component]!=own_model['metadata']['configurations'][component]:raise ValueError('Arm settings mismatch')
            if old_model['metadata']['fit_ids']!=own_model['metadata']['fit_ids']:raise ValueError('Arm training IDs mismatch')
            model_checks.append({'dataset':ds,'method':method,'same_settings':True,'same_training_ids':True})
        for dataset,role in COHORTS:
            if dataset!=ds:continue
            old_paths={int(p.stem.split('_')[-1]):p for p in (original/role).glob('profile_*.npz')}
            own_paths={int(p.stem.split('_')[-1]):p for p in (own/role).glob('profile_*.npz')}
            if not old_paths or old_paths.keys()!=own_paths.keys() or set(old_paths)!=set(roles[role]):raise ValueError('Missing paired complete cohort')
            for pid in sorted(old_paths):
                with np.load(old_paths[pid],allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
                with np.load(own_paths[pid],allow_pickle=False) as z:b={k:z[k].copy() for k in z.files}
                raw_path=(cache/cache_profiles[pid]['path']).resolve()
                if not raw_path.is_relative_to(cache.resolve()):raise ValueError('Unsafe cache path')
                with np.load(raw_path,allow_pickle=False) as z:target=z['y'].copy()
                exact(a['index'],np.arange(300,len(target),dtype=np.int64),'Full retained row coverage')
                exact(a['target'],target[a['index']],'Cached target mismatch')
                exact(a['index'],b['index'],'Portable arm endpoint mismatch')
                exact(a['target'],b['target'],'Portable arm target mismatch')
                s,p,h=profile_metrics_pair(ds,role,pid,a['index'],a['target'],
                    {m:a['prediction__'+m] for m in METHODS},{m:b['prediction__'+m] for m in METHODS})
                segments+=s;pairs+=p;whole+=h
    summaries,whole_summary=aggregate_pairs(pairs,whole)
    out.mkdir(parents=True,exist_ok=False)
    for name in ('analysis','figures','build'):(out/name).mkdir()
    smoke=run['mode']=='smoke'
    for rows in (segments,pairs,whole,summaries,whole_summary):
        for row in rows:row['smoke_only']=smoke
    save_metrics(out,segments,pairs,whole,summaries,whole_summary)
    result={'schema':'review-initialization-results-v1','status':'COMPLETE','input_mode':'portable_run',
        'run_key':run['key'],'execution_mode':run['mode'],'smoke_only':run['mode']=='smoke',
        'new_fits':0,'new_predictions':0,'new_formal_tests':0,'all_data_previously_exposed':True,
        'paired_profiles':len(whole)//len(METHODS),'retained_rows_per_arm':sum(r['n'] for r in whole if r['method']==METHODS[0]),
        'model_settings_checks':model_checks,'whole_summary':whole_summary,'fifth_summary':summaries,
        'slowest_filter_difference_after_300_rows':(1.-2./4097.)**300,
        'interpretation':'Same-model own minus original differences across retained-row fifths; no physical-time or initial-state recovery claim.',
        'historical_fallback_used':False,'run_manifest_sha256':sha(root/'run.json'),
        'script_sha256':sha(__file__),'sources':[{'path':k,'sha256':v} for k,v in sorted(sources.items())]}
    dump(out/'analysis/initialization-results.json',result)
    return summaries,result

def plot(rows,out,smoke=False):
    os.environ.setdefault('MPLCONFIGDIR',str(out/'build/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(7.2,5.2),sharex=True)
    colors={'linear_residual':'#0072B2','direct200':'#D55E00','d9':'#009E73'}
    markers={'linear_residual':'o','direct200':'s','d9':'^'}
    for idx,(ds,role) in enumerate(COHORTS):
        ax=axes.flat[idx]
        selected=[r for r in rows if r['dataset']==ds and r['role']==role]
        for method in METHODS:
            current=sorted((r for r in selected if r['method']==method),key=lambda r:int(r['fifth']))
            ax.plot([int(r['fifth']) for r in current],[float(r['macro_difference']) for r in current],
                    marker=markers[method],color=colors[method],lw=1.4,ms=4.5,label=LABELS[method])
        count=int(selected[0]['profiles'])
        ax.set_title(('IPMSM' if ds=='ipmsm' else 'Paderborn')+'\n'+ROLE[role]+f' (n={count})',fontsize=8.5)
        ax.axhline(0,color='black',lw=.8);ax.grid(axis='y',alpha=.18);ax.set_xticks([1,2,3,4,5]);ax.set_xlim(.75,5.25)
        ax.tick_params(axis='x',labelbottom=True)
        ax.set_xlabel('Retained-row fifth')
        if idx in (0,3):ax.set_ylabel('Macro-RMSE change (°C)')
    # Preserve comparable scales within each dataset, not across unlike regimes.
    for ds,indices in [('ipmsm',[0,1,2]),('paderborn',[3,4])]:
        values=[float(r['macro_difference']) for r in rows if r['dataset']==ds]
        lower=min(0.,min(values));upper=max(0.,max(values));padding=max(.03,.12*(upper-lower))
        for i in indices:axes.flat[i].set_ylim(lower-padding,upper+padding)
    axes.flat[5].axis('off')
    handles,labels=axes.flat[0].get_legend_handles_labels()
    axes.flat[5].legend(handles,labels,loc='upper left',frameon=False,fontsize=8.5,
                        handlelength=1.5,handletextpad=.5,borderaxespad=0.,bbox_to_anchor=(-.02,.91))
    axes.flat[5].text(0.,.53,'Change: channel-wise\nminus ambient.\nNegative: lower RMSE.\n\nShared y scale per dataset.\nFifths count rows, not time.',transform=axes.flat[5].transAxes,va='top',fontsize=8.5,linespacing=1.4)
    fig.subplots_adjust(left=.11,right=.985,bottom=.10,top=.89,wspace=.37,hspace=.69)
    if smoke:fig.text(.5,.012,'SMOKE TEST: reduced rows, trees, and epochs; not paper results',ha='center',fontsize=8,color='#555555')
    for suffix in ('pdf','png'):
        fig.savefig(out/f'figures/figure6-initialization-segments.{suffix}',dpi=250)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True,help='Completed portable run; no historical archive fallback.')
    p.add_argument('--output',type=Path,required=True,help='New output directory outside --run-dir.')
    args=p.parse_args();out=args.output.resolve()
    rows,result=compute_from_run(args.run_dir,out)
    plot(rows,out,result['smoke_only'])
    caption=('Same-model initialization differences over retained-row fifths. Each point is the equally weighted mean, across all profiles in the named cohort, of channel-wise minus ambient-initialized RMSE for the same predictor. Negative values favor channel-wise initialization. All predictors were separately refitted under fixed settings. Y-axis ranges are shared within each dataset. The two-profile IPMSM historical comparison is descriptive. Fifths are formed from retained row counts rather than elapsed physical time, and the results do not identify an unknown physical rotor starting temperature.')
    (out/'analysis/initialization-caption.txt').write_text(caption+'\n',encoding='utf-8')
    files=sorted([*out.glob('analysis/initialization-*'),*out.glob('figures/figure6-*')])
    dump(out/'analysis/initialization-output-manifest.json',{'files':[{'path':p.relative_to(out).as_posix(),'sha256':sha(p)} for p in files if p.name!='initialization-output-manifest.json']})
    print(json.dumps({'status':'COMPLETE','paired_profiles':result['paired_profiles'],
        'fifth_summary_cells':len(rows),'mode':result['execution_mode'],'output':str(out)}))

if __name__=='__main__':main()
