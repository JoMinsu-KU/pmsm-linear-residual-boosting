"""NumPy-only positive RC core, definition identical to frozen C12.

Separated from C12's sklearn imports for the existing CUDA environment.
This is shared scientific code, not an independent audit implementation.
"""
import json
import time
import numpy as np

FIT = [0, 2, 3, 5, 7, 8, 11, 14]
LAMBDAS = [.0001, .01, 1.]


def indices(record):
    return np.arange(300, len(record['y']), 5)


def scaling(records, ids):
    ix = {p: indices(records[p]) for p in ids}
    instant = np.concatenate([records[p]['instant'][ix[p]] for p in ids])
    ys = np.concatenate([records[p]['y'][ix[p]] for p in ids])
    rms_i = max(float(np.sqrt(np.mean(instant[:, 0]**2))), 1e-8)
    rms_w = max(float(np.sqrt(np.mean(instant[:, 3]**2))), 1e-8)
    ewma = np.concatenate([records[p]['ewma'][ix[p]] for p in ids])
    return dict(ids=list(ids), qscale=np.tile([rms_i**2, rms_w**2, rms_i*rms_w], 5).tolist(),
                y_mid=float(np.median(ys)), y_scale=max(float(np.percentile(ys,75)-np.percentile(ys,25)), 1.),
                ewma_mean=ewma.mean(0).tolist(), ewma_scale=np.maximum(ewma.std(0),1e-8).tolist())


def features(record, scale):
    return np.column_stack([record['boundary'], record['heat']/scale['qscale'], record['instant']])


def design(record, scale):
    return np.column_stack([(record['boundary']-scale['y_mid'])/scale['y_scale'], record['heat']/scale['qscale']])


def predict(model, record):
    return model['scale']['y_mid']+model['scale']['y_scale']*(design(record,model['scale'])@model['coef'])


def positive_qp(q, r, constant):
    np.linalg.cholesky(q)
    n=len(r); e=np.r_[np.ones(10),np.zeros(n-10)]
    theta=np.zeros(n); theta[0]=1.; free=[0]
    tol=2e-10*(1.+float(np.max(np.abs(r))))
    for iteration in range(1,1001):
        f=np.array(sorted(free),dtype=int); qf,ef,rf=q[np.ix_(f,f)],e[f],r[f]
        solved=np.linalg.solve(qf,np.column_stack([rf,ef])); denom=float(ef@solved[:,1])
        if denom<=0: raise ValueError('QP face lost feasible boundary coefficient')
        nu=float((ef@solved[:,0]-1.)/denom)
        candidate=np.zeros(n); candidate[f]=solved[:,0]-nu*solved[:,1]
        negative=[j for j in f if candidate[j]<-1e-12]
        if negative:
            alpha,hit=min((theta[j]/(theta[j]-candidate[j]),int(j)) for j in negative)
            theta+=alpha*(candidate-theta); theta[hit]=0.; free.remove(hit); continue
        theta=candidate; grad=q@theta-r+nu*e
        inactive=[j for j in range(n) if j not in free]
        if inactive:
            add=min(inactive,key=lambda j:(grad[j],j))
            if grad[add]<-tol: free.append(add); continue
        break
    else: raise ValueError('QP active-set iteration limit')
    grad=q@theta-r+nu*e; mu=np.maximum(grad,0.)
    dual_argument=r-nu*e+mu; z=np.linalg.solve(q,dual_argument)
    primal=float(.5*theta@q@theta-r@theta+constant)
    dual=float(constant-nu-.5*dual_argument@z)
    gap=float(.5*(theta-z)@q@(theta-z)+mu@theta+nu*(1.-e@theta))
    equality=float(abs(e@theta-1.)); gap_tol=1e-8*(1.+abs(primal))
    cert=dict(iterations=iteration,primal=primal,dual=dual,direct_gap=primal-dual,
        stable_gap=gap,gap_tolerance=gap_tol,equality_residual=equality,min_coefficient=float(theta.min()),
        nu=nu,mu=mu.tolist(),stationarity=float(np.max(np.abs(grad-mu))),
        complementarity=float(np.max(np.abs(mu*theta))),active_coefficients=[int(j) for j in np.flatnonzero(theta>1e-10)])
    if equality>1e-10 or theta.min()<-1e-10 or gap<-1e-10 or gap>gap_tol or abs(primal-dual-gap)>1e-8*(1.+abs(primal)):
        raise ValueError('QP numerical qualification failed: '+json.dumps(cert))
    return theta,cert


def fit(records, ids, lam, clock):
    clock.check(); started=time.monotonic(); scale=scaling(records,ids)
    xs=[design(records[p],scale)[indices(records[p])] for p in ids]
    ys=[(records[p]['y'][indices(records[p])]-scale['y_mid'])/scale['y_scale'] for p in ids]
    x,y=np.concatenate(xs),np.concatenate(ys)
    w=np.concatenate([np.full(len(v),1./(len(ids)*len(v))) for v in ys])
    q=x.T@(w[:,None]*x)+lam*np.eye(25); r=x.T@(w*y)
    coef,cert=positive_qp(q,r,float(.5*np.dot(w,y*y))); clock.check()
    return dict(family='positive',lam=lam,scale=scale,coef=coef.tolist(),certificate=cert,seconds=time.monotonic()-started)


def select(records, ids, clock):
    if len(ids)<2: raise ValueError('At least two profiles required for inner selection')
    table,folds=[],[]
    for lam in LAMBDAS:
        errors=[]
        for held in ids:
            train=[p for p in ids if p!=held]; model=fit(records,train,lam,clock)
            ix=indices(records[held]); prediction=predict(model,records[held])[ix]
            error=float(np.sqrt(np.mean((prediction-records[held]['y'][ix])**2)))
            errors.append(error); folds.append(dict(lam=lam,held_out=held,model=model,rmse=error))
        table.append(dict(lam=lam,profile_rmse=errors,macro_rmse=float(np.mean(errors))))
    chosen=min(range(3),key=lambda i:(table[i]['macro_rmse'],i)); lam=LAMBDAS[chosen]
    final=fit(records,ids,lam,clock)
    return final,dict(family='positive',selected_lambda=lam,cv=table,folds=folds,
                      all_fits=3*len(ids)+1,fit_ids=list(ids))
