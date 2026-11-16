"""Metadata/hash-only original legacy timing config builder; never loads arrays/models."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = dict(path='linear-followup-20261111/protocols/c18-review-followup-v001-frozen.md',
                sha256='00aee40005399af1df692671e757d998f937eeb939817a5251b6ff478a92de0d')
SOURCE = ROOT/'linear-followup-20261111/code/c18_streaming_cost_v001.py'
METHODS = ['D','P','L','H','direct200','capacity','free25_residual','free25_half','mlp_mse_seed1']


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def local(p):
    v=Path(p);v=(v if v.is_absolute() else ROOT/v).resolve()
    if not v.is_relative_to(ROOT):raise ValueError('Campaign-local input')
    return v


def ref(p,selector=None):
    p=local(p);v=dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p))
    if selector is not None:v['selector']=selector
    return v


def read(p):return json.loads(local(p).read_text(encoding='utf-8-sig'))


def pin(e):
    p=local(e['path'])
    if sha(p)!=e['sha256']:raise ValueError('Changed original artifact: '+str(p))
    return p


def build(dataset,mode):
    riskpath=ROOT/f'followup-risk-20261110/configs/{dataset}-risk-v001.json'
    risk=read(riskpath);role=risk['role_inputs'][0];e=role['records'][0]
    assert e['id']==role['ids'][0]
    lockpath=ROOT/f'runs/c15-paper-{dataset}-assemble-v001/model_lock.json';lock=read(lockpath)
    components={};lineage=[ref(riskpath),ref(lockpath),role['prediction_lock'],ref(SOURCE)]
    for name,job in [('d9',lock['selected_direct9']),('linear_residual_tree','linear33_insample_boost'),
                     ('direct200','direct_boost'),('free_residual_tree','unconstrained_rc_insample_boost')]:
        entry=lock['entries']['full__'+job];folder=local(entry['path']);receipt=folder/'receipt.json'
        if sha(receipt)!=entry['receipt_sha256']:raise ValueError('Original job receipt')
        md=folder/'model.json';m=read(md)
        if read(receipt)['files']['model.json']!=sha(md):raise ValueError('Original model metadata')
        pp=folder/m['pickle']['path']
        if sha(pp)!=m['pickle']['sha256']:raise ValueError('Original trusted model pickle')
        components[name]=dict(kind='tree',metadata=ref(md),pickle=ref(pp),scale=ref(md,['scale']))
        lineage.append(ref(receipt))
        if name=='linear_residual_tree':components['linear_prior']=dict(kind='linear',metadata=ref(md,['prior']))
        if name=='free_residual_tree':components['free_prior']=dict(kind='linear',metadata=ref(md,['prior']))
    cp=ROOT/f'linear-followup-20261111/runs/capacity-{dataset}-v001'
    caplock=read(cp/'prediction_lock.json');cm=read(cp/'model.json')
    capentry=next(v for v in caplock['records'] if v['role']==role['role'] and v['id']==e['id'])
    components['capacity']=dict(kind='tree',metadata=ref(cp/'model.json'),pickle=cm['pickle'],scale=ref(cp/'model.json',['scale']))
    lineage.append(ref(cp/'prediction_lock.json'))
    mp=ROOT/f'followup-risk-20261110/runs/{dataset}-mlp-mse-2026111701-v001'
    mlpmeta=read(mp/'model.json')
    if mlpmeta['loss']!='mse' or mlpmeta['seed']!=2026111701:raise ValueError('Fixed first MSE seed')
    mplock=read(mp/'prediction_lock.json')
    mpe=next(v for v in mplock['predictions'] if v['role']==role['role'] and v['profile']==e['id'])
    components['mlp']=dict(kind='mlp',metadata=ref(mp/'model.json'),weights=ref(mp/'weights_final.npz'))
    lineage.extend([ref(mp/'prediction_lock.json'),ref(mp/'model_lock.json')])
    audits=[role['prior_evaluation_audit'],ref(ROOT/f'runs/c15-paper-audit-{dataset}-selection-all-v001/report.json'),
            ref(ROOT/f'linear-followup-20261111/runs/capacity-{dataset}-audit-v002/report.json'),
            ref(ROOT/f'followup-risk-20261110/runs/{dataset}-mlp-audit-mse-2026111701-v001/report.json')]
    for a in audits:
        z=read(pin(a))
        if z.get('status')!='PASS' or z.get('failures'):raise ValueError('Prior independent PASS required')
    def expected(key,weight=1.,artifact=None):
        return dict(artifact=e['features'] if artifact is None else artifact,index_key='index',key=key,weight=weight)
    def method(name,terms,ex):return dict(name=name,terms=terms,expected=ex)
    d='prediction__direct9_fitcv_selected';l='prediction__linear33_insample_boost';f='prediction__unconstrained_rc_insample_boost'
    methods=[method('D',[['d9',1.]],[expected(d)]),
        method('P',[['linear_prior',1.]],[expected('prediction__linear33_ridge')]),
        method('L',[['linear_prior',1.],['linear_residual_tree',1.]],[expected(l)]),
        method('H',[['d9',.5],['linear_prior',.5],['linear_residual_tree',.5]],[expected(d,.5),expected(l,.5)]),
        method('direct200',[['direct200',1.]],[expected('prediction__direct_boost')]),
        method('capacity',[['capacity',1.]],[expected('prediction',artifact=capentry['prediction'])]),
        method('free25_residual',[['free_prior',1.],['free_residual_tree',1.]],[expected(f)]),
        method('free25_half',[['d9',.5],['free_prior',.5],['free_residual_tree',.5]],[expected(d,.5),expected(f,.5)]),
        method('mlp_mse_seed1',[['mlp',1.]],[expected(mpe['prediction_key'],artifact={k:mpe[k] for k in ('path','sha256')})])]
    assert [m['name'] for m in methods]==METHODS
    for item in lineage:pin(item)
    for c in components.values():
        for name in ('metadata','pickle','scale','weights'):
            if name in c:pin(c[name])
    for m in methods:
        for ee in m['expected']:pin(ee['artifact'])
    pin(e['features']);pin(PROTOCOL)
    return dict(schema='c18-streaming-cost-config-v001',protocol=PROTOCOL,producer_sha256=sha(SOURCE),
        builder_sha256=sha(__file__),no_concurrent_training_required=True,mode=mode,warmups=1,
        repetitions=2 if mode=='probe' else 5,lineage=lineage,audits=audits,
        case=dict(dataset=dataset,arm='original',role=role['role'],profile_id=e['id'],
                  selection_rule='first record in original calibration/development role order, before timing',
                  operating=e['features'],operating_key='operating',feature_reference=e['features']),
        components=components,methods=methods,
        optional_mlp_other_seed_weight_sizes=[dict(seed=s,weights=ref(ROOT/f'followup-risk-20261110/runs/{dataset}-mlp-mse-{s}-v001/weights_final.npz'),
            bytes=(ROOT/f'followup-risk-20261110/runs/{dataset}-mlp-mse-{s}-v001/weights_final.npz').stat().st_size,
            timed=False) for s in (2026111702,2026111703)])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',choices=['ipmsm','paderborn'],required=True)
    p.add_argument('--mode',choices=['probe','main'],required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=build(a.dataset,a.mode);out=local(a.output)
    if WORK_CONFIG not in out.parents:raise ValueError('New C18 config directory only')
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(path=str(out),sha256=sha(out),metadata_only=True,methods=METHODS)))


WORK_CONFIG=ROOT/'linear-followup-20261111/configs'
if __name__=='__main__':main()
