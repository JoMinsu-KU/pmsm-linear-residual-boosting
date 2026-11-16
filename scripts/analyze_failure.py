"""Reanalyze locked complete predictions; no model fitting or inference."""
from pathlib import Path
import argparse,csv,hashlib,json,math,os
from collections import defaultdict
import numpy as np

METHODS=('linear_prior','linear_residual','direct200','d9')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1048576),b''):h.update(part)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def dump(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def write_csv(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def variables(record):
    op=record['operating']
    return {'rotor_temperature':record['y'],'current_magnitude':np.hypot(op[:,0],op[:,1]),
            'absolute_speed':np.abs(op[:,4]),'coolant_temperature':op[:,5]}

def distribution(values):
    q=np.quantile(values,[.01,.05,.5,.95,.99])
    return dict(n=len(values),minimum=float(np.min(values)),q01=float(q[0]),q05=float(q[1]),
        median=float(q[2]),q95=float(q[3]),q99=float(q[4]),maximum=float(np.max(values)))

def support(values,fit):
    return {'below_fit_min_percent':float(100*np.mean(values<fit['minimum'])),
        'above_fit_max_percent':float(100*np.mean(values>fit['maximum'])),
        'outside_fit_minmax_percent':float(100*np.mean((values<fit['minimum'])|(values>fit['maximum']))),
        'outside_fit_q05_q95_percent':float(100*np.mean((values<fit['q05'])|(values>fit['q95'])))}

def metric_rows(ds,arm,role,pid,z,y):
    p,l,d=(z['prediction__'+m] for m in METHODS[:3]);e=p-y;c=l-p
    errors={m:z['prediction__'+m]-y for m in METHODS}
    mse={m:float(np.mean(v*v)) for m,v in errors.items()}
    cross=float(2*np.mean(e*c));energy=float(np.mean(c*c));change=mse['linear_residual']-mse['linear_prior']
    discrepancy=abs(change-cross-energy)
    if discrepancy>1e-10*max(1,mse['linear_prior'],mse['linear_residual']):raise ArithmeticError('Residual identity mismatch')
    row=dict(dataset=ds,arm=arm,role=role,profile=pid,n=len(y),
        **{m+'_rmse_c':math.sqrt(v) for m,v in mse.items()},
        **{m+'_bias_c':float(np.mean(v)) for m,v in errors.items()},
        prior_mse_c2=mse['linear_prior'],residual_mse_c2=mse['linear_residual'],
        correction_cross_term_c2=cross,correction_energy_c2=energy,residual_minus_prior_mse_c2=change,
        residual_minus_prior_rmse_c=math.sqrt(mse['linear_residual'])-math.sqrt(mse['linear_prior']),
        residual_minus_matched_rmse_c=math.sqrt(mse['linear_residual'])-math.sqrt(mse['direct200']),
        rowwise_squared_error_reduced_percent=float(100*np.mean(errors['linear_residual']**2<e**2)),
        correction_against_prior_error_percent=float(100*np.mean(e*c<0)),
        correction_with_prior_error_percent=float(100*np.mean(e*c>0)),
        opposing_but_overshooting_percent=float(100*np.mean((e*c<0)&(np.abs(c)>2*np.abs(e)))),
        correction_mean_c=float(np.mean(c)),correction_rms_c=math.sqrt(energy),identity_absolute_discrepancy=discrepancy)
    return row

def plot_examples(examples,output,smoke=False):
    os.environ.setdefault('MPLCONFIGDIR',str(output.parent/'build/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':9.5,'axes.labelsize':9,
                         'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(7.4,5.0),sharex='col')
    styles={'target':('Measured rotor','#222222','-',1.15),
        'linear_prior':('Ridge prior','#777777','--',1.),
        'linear_residual':('Linear residual boosting','#0072B2','-',1.0),
        'direct200':('Matched direct boosting','#D55E00',':',1.05)}
    rows=[]
    for column,example in enumerate(examples):
        z,y,row=example['arrays'],example['target'],example['metrics'];index=z['index']
        for key,(label,color,style,width) in styles.items():
            v=y if key=='target' else z['prediction__'+key]
            axes[0,column].plot(index,v,label=label,color=color,ls=style,lw=width)
            if key!='target':axes[1,column].plot(index,v-y,color=color,ls=style,lw=width)
        axes[0,column].set_title(('Paderborn' if row['dataset']=='paderborn' else 'IPMSM')+', profile '+str(row['profile'])+'\n'+('Median-near improvement' if column==0 else 'Largest historical degradation'))
        axes[1,column].axhline(0,color='black',lw=.6)
        axes[1,column].set_xlabel('Within-profile source-row index')
        for ax in axes[:,column]:ax.grid(alpha=.18);ax.set_xlim(int(index[0]),int(index[-1]))
        axes[1,column].text(.03,.04,'RMSE: ridge %.2f, residual %.2f, direct %.2f °C'%(row['linear_prior_rmse_c'],row['linear_residual_rmse_c'],row['direct200_rmse_c']),
            transform=axes[1,column].transAxes,fontsize=7.3,bbox=dict(facecolor='white',alpha=.9,edgecolor='none'))
        for i,j in enumerate(index):
            rows.append(dict(dataset=row['dataset'],arm=row['arm'],role=row['role'],profile=row['profile'],within_profile_row=int(j),target_c=float(y[i]),
                **{m+'_c':float(z['prediction__'+m][i]) for m in METHODS[:3]}))
    axes[0,0].set_ylabel('Rotor temperature (°C)');axes[1,0].set_ylabel('Prediction minus target (°C)')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1.0),ncol=2,frameon=False)
    fig.subplots_adjust(top=.82,bottom=.14 if smoke else .105,left=.09,right=.99,hspace=.2,wspace=.22)
    if smoke:fig.text(.5,.015,'SMOKE TEST: reduced data and training; not paper results',ha='center',fontsize=8,color='#555555')
    for ext in ('pdf','png'):fig.savefig(output/('figure5-temperature-trajectories.'+ext),dpi=240,bbox_inches='tight')
    plt.close(fig);write_csv(output/'figure5-plotted-data.csv',rows)

def aggregate_rows(profile_rows):
    aggregates=[];groups=defaultdict(list)
    for row in profile_rows:groups[(row['dataset'],row['arm'],row['role'])].append(row)
    for (ds,arm,role),rows in groups.items():
        aggregates.append(dict(dataset=ds,arm=arm,role=role,profiles=len(rows),n=sum(r['n'] for r in rows),
            **{m+'_macro_rmse_c':float(np.mean([r[m+'_rmse_c'] for r in rows])) for m in METHODS},
            residual_beats_prior_profiles=sum(r['residual_minus_prior_rmse_c']<0 for r in rows),
            residual_beats_matched_profiles=sum(r['residual_minus_matched_rmse_c']<0 for r in rows),
            mean_cross_term_c2=float(np.mean([r['correction_cross_term_c2'] for r in rows])),
            mean_correction_energy_c2=float(np.mean([r['correction_energy_c2'] for r in rows])),
            mean_mse_change_c2=float(np.mean([r['residual_minus_prior_mse_c2'] for r in rows])),
            mean_rowwise_improvement_percent=float(np.mean([r['rowwise_squared_error_reduced_percent'] for r in rows]))))
    return aggregates

def range_rows(raw,settings):
    ranges=[];profile_ranges=[]
    for ds,records in raw.items():
        fit=settings[ds]['roles']['fit'];var={pid:variables(r) for pid,r in records.items()}
        for variable in next(iter(var.values())):
            training=np.concatenate([var[p][variable][300::5] for p in fit]);reference=distribution(training)
            ranges.append(dict(dataset=ds,role='fit',scope='model_fitting_endpoints',variable=variable,profiles=len(fit),**reference))
            all_raw=np.concatenate([var[p][variable] for p in fit])
            ranges.append(dict(dataset=ds,role='fit',scope='complete_cached_training_profiles',variable=variable,profiles=len(fit),**distribution(all_raw)))
            for role,ids in settings[ds]['roles'].items():
                if role=='fit':continue
                values=np.concatenate([var[p][variable][300:] for p in ids])
                ranges.append(dict(dataset=ds,role=role,scope='retained_evaluation_endpoints',variable=variable,profiles=len(ids),**distribution(values),**support(values,reference)))
                for pid in ids:
                    values=var[pid][variable][300:]
                    profile_ranges.append(dict(dataset=ds,role=role,profile=pid,variable=variable,**distribution(values),**support(values,reference)))
    return ranges,profile_ranges

def selection(profile_rows):
    candidates=[r for r in profile_rows if r['dataset']=='paderborn' and r['arm']=='own_boundary' and r['role']=='confirmation']
    if not candidates:return [],None,'Paderborn historical cohort missing'
    median=float(np.median([r['residual_minus_matched_rmse_c'] for r in candidates]))
    improving=[r for r in candidates if r['residual_minus_matched_rmse_c']<0]
    failures=[r for r in profile_rows if r['dataset']=='ipmsm' and r['arm']=='own_boundary' and r['role']=='confirmation' and r['residual_minus_matched_rmse_c']>0]
    if not improving or not failures:return [],median,'No qualifying Paderborn improvement or IPMSM degradation under the stated selection rule'
    success=min(improving,key=lambda r:(abs(r['residual_minus_matched_rmse_c']-median),r['profile']))
    failure=min(failures,key=lambda r:(-r['residual_minus_matched_rmse_c'],r['profile']))
    return [success,failure],median,None

def run_diagnostics(run_dir,output,config_path=None):
    """Fresh-run adapter: reads only the supplied verified run; never archives."""
    run_dir=Path(run_dir).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError('Choose a new --output directory')
    if output.is_relative_to(run_dir):raise ValueError('Keep analysis output outside the immutable run directory')
    run=read(run_dir/'run.json');sources=[]
    binding={name:run[name] for name in ('schema','mode','datasets','numerical_runtime','device','files')}
    expected=hashlib.sha256(json.dumps(binding,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if run['key']!=expected:raise ValueError('Run binding mismatch: metadata changed')
    if run['schema']!='portable-study-run-v2':raise ValueError('Unsupported run schema')
    if run['mode'] not in ('smoke','paper'):raise ValueError('Unknown run mode')
    if not {'ipmsm','paderborn'}.issubset(run['datasets']):raise ValueError('Both study datasets are required')
    def stage(name):
        folder=run_dir/name
        if folder.is_symlink() or not folder.resolve().is_relative_to(run_dir):raise ValueError('Unsafe stage path')
        receipt=read(folder/'complete.json')
        expected=hashlib.sha256(json.dumps({'run':run['key'],'stage':name,'extra':None},sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if receipt['input_key']!=expected:raise ValueError('Stage binding mismatch: '+name)
        names=[r['path'] for r in receipt['files']]
        actual={p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file() and p.name!='complete.json'}
        if len(names)!=len(set(names)) or set(names)!=actual:raise ValueError('Stage file inventory differs: '+name)
        for entry in receipt['files']:
            p=folder/entry['path']
            if p.is_symlink() or not p.resolve().is_relative_to(folder.resolve()) or sha(p)!=entry['sha256']:
                raise ValueError('Stage file hash/path mismatch: '+str(p))
        sources.append({'stage':name,'complete_manifest_sha256':sha(folder/'complete.json')})
        return folder
    report=stage('report');metadata=read(report/'metadata.json');config=metadata['configuration']
    for ds in ('ipmsm','paderborn'):
        spec=config['datasets'][ds]
        if any(spec.get(name)!=value for name,value in [('warmup',300),('train_stride',5),('evaluation_stride',1)]):
            raise ValueError('Unsupported warmup/stride configuration: '+ds)
    if config_path is not None:
        config_path=Path(config_path)
        if sha(config_path)!=run['files']['configs/study.json'] or read(config_path)!=config:
            raise ValueError('Explicit configuration differs from the fitted run')
    settings=config['datasets'];raw={};profile_rows=[];data_by_key={}
    for ds in ('ipmsm','paderborn'):
        cache=stage('cache/'+ds);cachemeta=read(cache/'dataset.json');records={}
        if cachemeta['dataset']!=ds or cachemeta['roles']!=settings[ds]['roles']:raise ValueError('Cache partition differs')
        for item in cachemeta['profiles']:
            p=(cache/item['path']).resolve()
            if not p.is_relative_to(cache.resolve()):raise ValueError('Unsafe cache path')
            with np.load(p,allow_pickle=False) as z:records[int(item['id'])]={k:z[k].copy() for k in z.files}
        expected_ids={int(pid) for ids in settings[ds]['roles'].values() for pid in ids}
        if set(records)!=expected_ids or len(records)!=len(cachemeta['profiles']):raise ValueError('Incomplete or duplicate cached profile set')
        raw[ds]=records
        for arm in ('original','own_boundary'):
            evaluated=stage(f'evaluation/{ds}/{arm}')
            for role,ids in settings[ds]['roles'].items():
                if role=='fit':continue
                actual_ids={int(p.stem.split('_')[-1]) for p in (evaluated/role).glob('profile_*.npz')}
                if actual_ids!=set(ids):raise ValueError('Incomplete evaluation profile set')
            for role,ids in settings[ds]['roles'].items():
                if role=='fit':continue
                for pid in ids:
                    with np.load(evaluated/role/f'profile_{pid}.npz',allow_pickle=False) as stored:z={k:stored[k].copy() for k in stored.files}
                    index=np.arange(300,len(records[pid]['y']),dtype=np.int64)
                    np.testing.assert_array_equal(index,z['index'])
                    np.testing.assert_array_equal(records[pid]['y'][index],z['target'])
                    for method in METHODS:
                        if z['prediction__'+method].shape!=index.shape or not np.isfinite(z['prediction__'+method]).all():
                            raise ValueError('Invalid prediction shape/values')
                    row=metric_rows(ds,arm,role,pid,z,z['target']);profile_rows.append(row)
                    data_by_key[(ds,arm,role,pid)]={'arrays':z,'target':z['target'],'metrics':row}
    ranges,profile_ranges=range_rows(raw,settings)
    selected,median,unavailable=selection(profile_rows)
    smoke=run['mode']=='smoke'
    if run['mode'] not in ('smoke','paper'):raise ValueError('Unknown run mode')
    output.mkdir(parents=True,exist_ok=False)
    for rows in (profile_rows,ranges,profile_ranges):
        for row in rows:row['smoke_only']=smoke
    cohorts=aggregate_rows(profile_rows)
    for row in cohorts:row['smoke_only']=smoke
    write_csv(output/'failure-profile-corrections.csv',profile_rows)
    write_csv(output/'failure-cohort-corrections.csv',cohorts)
    write_csv(output/'failure-operating-ranges.csv',ranges)
    write_csv(output/'failure-profile-support.csv',profile_ranges)
    if selected:
        figures=output/'figures';figures.mkdir()
        plot_examples([data_by_key[(r['dataset'],r['arm'],r['role'],r['profile'])] for r in selected],figures,smoke=smoke)
    result={'schema':'portable-failure-reanalysis-v1','status':'COMPLETE','run_key':run['key'],
        'mode':run['mode'],'smoke_only':smoke,'new_fits':0,'new_predictions':0,'historical_fallback_used':False,
        'profiles':len(profile_rows),'cohorts':cohorts,'selected_examples':selected,
        'success_selection_full_cohort_median_difference_c':median,'figure_unavailable_reason':unavailable,
        'range_scope':'Actual fitting endpoints versus retained evaluation rows from the supplied caches; smoke caches are truncated.',
        'units':{'rotor_temperature':'degC','current_magnitude':'source current units','absolute_speed':'source speed units','coolant_temperature':'degC'},
        'selection_rule':'Paderborn improving record nearest historical cohort median LR-minus-matched RMSE, and largest positive IPMSM historical degradation; ties by lower profile ID.',
        'inference':'Descriptive exposed-data reanalysis; marginal range checks do not establish joint/history coverage or causal mechanisms.',
        'run_manifest_sha256':sha(run_dir/'run.json'),
        'source_manifests':sources,'script_sha256':sha(__file__)}
    dump(output/'failure-results.json',result)
    print(json.dumps({'status':'COMPLETE','mode':run['mode'],'profiles':len(profile_rows),'figure_unavailable_reason':unavailable,'output':str(output)},indent=2))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True,help='Completed portable run; no historical archive fallback.')
    parser.add_argument('--output',type=Path,required=True,help='New output directory outside --run-dir.')
    parser.add_argument('--config',type=Path,help='Optional explicit config; must match the supplied run exactly.')
    args=parser.parse_args()
    run_diagnostics(args.run_dir,args.output,args.config)

if __name__=='__main__':main()
