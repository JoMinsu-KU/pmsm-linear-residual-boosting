"""Evaluation of newly fitted predictors, retaining every configured profile."""
from pathlib import Path
import numpy as np
from . import features,metrics,models,neural
from .io import dump_json

METHOD_KEYS={"selected_direct":"d9","matched_direct":"direct200","matched_half":"direct200_half",
             "history_prior":"free_prior","history_residual":"free_residual","history_half":"free_half"}

def evaluate(records,bundle,networks,roles,dataset,arm,out,check=lambda:None,device="cuda"):
    out=Path(out);accuracy=[];profiles=[];diagnostics=[];segments=[]
    for role,ids in roles.items():
        if role=="fit":continue
        role_rows=[];role_out=out/role;role_out.mkdir()
        for pid in ids:
            check();record=records[int(pid)];ix=features.evaluation_index(record)
            truth=np.asarray(record["target"])[ix]
            values={METHOD_KEYS.get(k,k):v[ix] for k,v in models.predict_cpu(bundle,record).items()}
            if networks:
                x=features.neural_features(record,bundle["scale"])
                for seed,model in networks.items():
                    values[f"mse_mlp_{seed}"]=neural.predict_neural(model,x[ix],device=device,
                        batch_size=4096,backend="torch")
            np.savez_compressed(role_out/f"profile_{pid}.npz",index=ix,target=truth,
                **{"prediction__"+k:v for k,v in values.items()})
            common={"dataset":dataset,"arm":arm,"role":role,"profile":int(pid)}
            for method,prediction in values.items():
                row={**common,"method":method,**metrics.profile_metrics(truth,prediction)}
                profiles.append(row);role_rows.append(row)
                for fifth in range(1,6):
                    lo=len(ix)*(fifth-1)//5;hi=len(ix)*fifth//5
                    part=np.arange(lo,hi)
                    if len(part):
                        segments.append({**common,"method":method,"fifth":fifth,
                            "first_source_index":int(ix[lo]),"last_source_index":int(ix[hi-1]),
                            **metrics.profile_metrics(truth[part],prediction[part])})
            diagnostic={**common,"ensemble":metrics.ensemble_decomposition(truth,values["d9"],values["linear_residual"]),
                "residual":metrics.residual_decomposition(truth,values["linear_prior"],values["linear_residual"])}
            if dataset!="rwth":
                delta=float(record["operating"][0,5]-record["operating"][0,6])
                a=1.-2./(np.asarray([16,64,256,1024,4096])+1.)
                diagnostic["coolant_minus_ambient_start"]=delta
                diagnostic["boundary_history_rms_difference"]=float(np.sqrt(np.mean((np.power(a[None,:],ix[:,None])*delta)**2)))
            diagnostics.append(diagnostic)
        for method in sorted({r["method"] for r in role_rows}):
            rows=[r for r in role_rows if r["method"]==method]
            accuracy.append({"dataset":dataset,"arm":arm,"role":role,"method":method,**metrics.aggregate_profiles(rows)})
    dump_json(out/"evaluation.json",{"accuracy":accuracy,"profiles":profiles,
        "diagnostics":diagnostics,"temporal_fifths":segments,
        "source":"Fresh predictions from this run's fitted artifacts; no historical metric fallback.",
        "exposure":"All configured evaluation records were historically exposed."})
    return accuracy,profiles
