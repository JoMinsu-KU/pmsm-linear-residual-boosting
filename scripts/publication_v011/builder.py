"""Saved-data drawing primitives; adapted from the manuscript figure generator.

The original coordinates, values and calculations are retained. Directory paths
are portable; unused historical receipt dependencies and entry point are omitted.
The v011 wrapper applies the final styles before exporting.
"""
from pathlib import Path
import csv, hashlib, json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.ticker import MultipleLocator, ScalarFormatter
ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/"reference/v011/data"
OUT=ROOT/"figures"
REVIEW=ROOT/"review"
BLUE="#0072B2"; ORANGE="#D55E00"; TEAL="#009E73"; PURPLE="#CC79A7"; GRAY="#555555"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.labelsize":8.5,
 "axes.titlesize":9,"xtick.labelsize":8,"ytick.labelsize":8,"legend.fontsize":8,
 "axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42,"ps.fonttype":42,
 "svg.fonttype":"none","svg.hashsalt":"c18-access-v002","axes.linewidth":.65})
def read(p):return json.loads(p.read_text(encoding="utf-8-sig"))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return {"path":p.relative_to(ROOT.parent).as_posix(),"sha256":sha(p),"bytes":p.stat().st_size}
SOURCES=[
 OLD/"paderborn-own-evaluation-source.json",
 OLD/"cost-comparison-source.json",
 OLD/"own-controls-comparison.json",
 OLD/"S2-own-profile-all-methods.csv",
]
SOURCE_HASHES={str(p):sha(p) for p in SOURCES}
NUMERIC={}; FIGS=[]
def save(fig,stem):
    fig.canvas.draw()
    # Fixed physical dimensions: no bbox_inches="tight" rescaling.
    w,h=map(float,fig.get_size_inches())
    for ext in ("pdf","svg","png"):
        fig.savefig(OUT/f"{stem}.{ext}",dpi=300,facecolor="white",metadata={"Creator":"C18 saved-result layout"})
    FIGS.append({"stem":stem,"width_inches":w,"height_inches":h,
      "minimum_font_points":8,"files":[ref(OUT/f"{stem}.{e}") for e in ("pdf","svg","png")]})
    plt.close(fig)
def box(ax,x,y,w,h,text,color=GRAY,fill="#F7F8FA",fs=8.5,bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.025,rounding_size=0.05",
      facecolor=fill,edgecolor=color,linewidth=.9))
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=fs,
      color=color,fontweight="bold" if bold else "normal",linespacing=1.25)
def arrow(ax,a,b,color=GRAY):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=8,linewidth=.9,color=color))
def method():
    fig=plt.figure(figsize=(7.1,2.5));ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,7.1),ylim=(0,2.5));ax.axis("off")
    ax.add_patch(FancyBboxPatch((.08,1.77),6.94,.61,boxstyle="round,pad=0.025",facecolor="#F2F5F8",edgecolor="#C8D0D9",linewidth=.7))
    ax.text(.18,2.22,"Training",fontsize=9.3,fontweight="bold",va="center",color=GRAY)
    for x,t in [(1.25,r"Fit $\hat T_{\rm lin}$ on $(x,y)$"),(3.03,r"Form $y-\hat T_{\rm lin}(x)$"),(4.84,r"Fit $\hat r$ to residuals")]:
        box(ax,x,2.05,1.42,.26,t,BLUE,fill="white",fs=8.3)
    arrow(ax,(2.71,2.18),(2.98,2.18),BLUE);arrow(ax,(4.49,2.18),(4.79,2.18),BLUE)
    ax.text(1.26,1.88,r"Fit $\hat T_{\rm dir}$ directly on $(x,y)$; freeze predictors before held-profile inference.",fontsize=8.2,va="center")
    ax.text(.10,1.54,"Inference: one incoming row; shared causal history update",fontsize=8.7,fontweight="bold",color=GRAY)
    box(ax,.08,.63,1.07,.65,"Available\noperating inputs\n(no target T)",fs=8.2)
    box(ax,1.44,.63,1.18,.65,"Causal history\n33 features",color=GRAY,fs=8.7)
    arrow(ax,(1.19,.955),(1.40,.955))
    ax.plot([2.65,2.80,2.80],[.955,.955,.32],color=GRAY,lw=.9)
    ax.plot([2.80,2.80],[.955,1.20],color=GRAY,lw=.9)
    for y,t,c in [(1.01,"Ridge regression",BLUE),(.54,"Residual boosting",BLUE),(.10,"Direct boosting",GRAY)]:
        box(ax,3.03,y,1.03,.38,t,color=c,fs=8.4);arrow(ax,(2.80,y+.19),(2.99,y+.19),c)
    ax.add_patch(Circle((4.43,.96),.13,facecolor="white",edgecolor=BLUE,lw=1))
    ax.text(4.43,.96,"+",ha="center",va="center",fontsize=11,color=BLUE)
    arrow(ax,(4.10,1.20),(4.32,1.01),BLUE);arrow(ax,(4.10,.73),(4.32,.91),BLUE)
    box(ax,4.80,.75,.86,.44,r"$\hat T_{\rm LR}$",color=BLUE,fill="#EAF4FA",fs=11,bold=True)
    arrow(ax,(4.57,.96),(4.76,.97),BLUE)
    box(ax,5.99,.69,1.02,.56,r"$\hat T_{\rm avg}$",color=ORANGE,fill="#FDF1EA",fs=11)
    arrow(ax,(5.70,.97),(5.95,.97),BLUE)
    ax.plot([4.10,6.50],[.29,.29],color=GRAY,lw=.9);arrow(ax,(6.50,.29),(6.50,.65))
    ax.text(5.60,.10,"Fixed 1:1 average",fontsize=8,ha="center",color=GRAY)
    save(fig,"figure1-method-overview")
def accuracy(p,cost):
    panel=next(x for x in cost["panels"] if x["dataset"]=="paderborn" and x["arm"]=="own")
    names={"D":"d9","L":"linear_residual","H":"linear_half","direct200":"direct200","capacity":"capacity_direct",
      "free25_residual":"free_residual","free25_half":"free_half","mlp_mse_seed1":"mse_mlp_2026111701"}
    labels={"D":"Selected direct boosting","L":"Linear residual boosting","H":"Equal-weight ensemble",
      "direct200":"Matched direct boosting (200 trees)","capacity":"Expanded direct boosting",
      "free25_residual":"History-only residual boosting","free25_half":"History-only ensemble",
      "mlp_mse_seed1":"Neural network (timed seed)"}
    colors={"D":GRAY,"L":BLUE,"H":ORANGE,"direct200":GRAY,"capacity":GRAY,
      "free25_residual":PURPLE,"free25_half":PURPLE,"mlp_mse_seed1":TEAL}
    # Label coordinates in display points fixed prospectively for layout only.
    offsets=[{"D":(7,8),"L":(-19,9),"H":(8,-5),"direct200":(-45,11),"capacity":(8,-12),
      "free25_residual":(-57,-16),"free25_half":(-71,-16),"mlp_mse_seed1":(7,-14)},
      {"D":(6,-13),"L":(-16,-17),"H":(8,-3),"direct200":(-48,9),"capacity":(6,10),
      "free25_residual":(-59,11),"free25_half":(-73,14),"mlp_mse_seed1":(7,7)}]
    markers={"D":"o","L":"o","H":"s","direct200":"^","capacity":"D",
      "free25_residual":"P","free25_half":"h","mlp_mse_seed1":"*"}
    fig,axs=plt.subplots(1,2,figsize=(7.1,4.0),sharey=True)
    data=[]
    for j,(ax,role,title) in enumerate(zip(axs,["development","confirmation"],["(a) Development · 14 profiles","(b) Historical evaluation · 14 profiles"])):
        c=next(x for x in p["roles"] if x["role"]==role)
        for name,key in names.items():
            x=panel["methods"][name]["median_us"];y=c["methods"][key]["macro_rmse"]
            data.append({"role":role,"method":name,"accuracy_key":key,"median_us":x,"macro_rmse":y})
            ax.scatter(x,y,color=colors[name],s=43 if name in ("L","H") else 32,zorder=3,
              marker=markers[name],label=labels[name],edgecolors="white",linewidths=.35)
        ax.set_xscale("log");ax.set_xlim(8,6500);ax.set_ylim(2.25,3.68)
        ax.set_xticks([10,100,1000],["10","100","1,000"])
        ax.grid(alpha=.22,lw=.5);ax.set_title(title,loc="left",pad=9)
        ax.set_xlabel("CPU median latency (µs/row, log scale)")
    axs[0].set_ylabel("Profile-macro RMSE (°C)")
    fig.subplots_adjust(left=.085,right=.985,bottom=.37,top=.92,wspace=.13)
    handles,legend_labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc="lower center",bbox_to_anchor=(.53,.025),ncol=2,
      frameon=False,columnspacing=2.1,handletextpad=.6,labelspacing=.65,fontsize=8)
    NUMERIC["figure2"]=data;save(fig,"figure2-accuracy-latency")
def paired(p):
    fig,axs=plt.subplots(2,1,figsize=(3.45,4.4),sharey=True)
    rowscsv=list(csv.DictReader((OLD/"S2-own-profile-all-methods.csv").open(encoding="utf-8-sig",newline="")))
    data=[]
    for ax,role,title in zip(axs,["development","confirmation"],["(a) Development · 14 profiles","(b) Historical evaluation · 14 profiles"]):
        rows=[x for x in p["per_profile"] if x["role"]==role]
        assert len(rows)==14
        vals=[];ids=[]
        for x in rows:
            a=x["methods"]["linear_residual"]["rmse"];b=x["methods"]["direct200"]["rmse"]
            for method,value in [("linear_residual",a),("direct200",b)]:
                refrow=next(z for z in rowscsv if z["dataset"]=="paderborn" and z["role"]==role and int(z["profile"])==x["profile"] and z["method"]==method)
                assert float(refrow["rmse"])==value
            vals.append(a-b);ids.append(x["profile"])
            data.append({"role":role,"profile_id":x["profile"],"L_rmse":a,"direct200_rmse":b,"difference":a-b})
        xs=np.arange(14);ax.bar(xs,vals,color=[BLUE if v<0 else ORANGE for v in vals],width=.72)
        ax.axhline(0,lw=.7,color="black");ax.set_xticks(xs,[str(v) for v in ids]);ax.set_xlim(-.65,13.65)
        ax.tick_params(axis="x",length=2,pad=2);ax.grid(axis="y",alpha=.2,lw=.5);ax.set_axisbelow(True)
        ax.set_title(title,loc="left",pad=5,fontsize=8.5);ax.set_xlabel("Profile ID (fixed order)",labelpad=4)
        ax.set_ylabel("RMSE difference (°C)",labelpad=4)
    fig.subplots_adjust(left=.20,right=.985,top=.935,bottom=.16,hspace=.62)
    fig.text(.5,.025,"Negative favors linear residual boosting.",ha="center",fontsize=8)
    NUMERIC["figure3"]=data;save(fig,"figure3-paired-profile-errors")
def initialization(comp):
    fig,axs=plt.subplots(3,1,figsize=(3.45,4.7),sharex=True)
    configs=[("ipmsm","confirmation","(a) IPMSM historical evaluation (2)"),("paderborn","development","(b) Paderborn development (14)"),("paderborn","confirmation","(c) Paderborn historical evaluation (14)")]
    keys=["d9","linear_residual","linear_half","free_half","mlp_seed_metric_mean"]
    labels=["Selected direct","Linear residual","Equal-weight","History-only avg.","Neural mean"]
    cols=[GRAY,BLUE,ORANGE,PURPLE,TEAL];data=[]
    for ax,(ds,role,title) in zip(axs,configs):
        c=next(z for z in comp["cohorts"] if z["dataset"]==ds and z["role"]==role)
        vals=[]
        for key in keys:
            q=c[key] if key=="mlp_seed_metric_mean" else c["methods"][key]
            a=q["original"]["macro_rmse"];b=q["own_boundary"]["macro_rmse"];v=100*(b-a)/a;vals.append(v)
            if key=="mlp_seed_metric_mean":
                for arm in ("original","own_boundary"):
                    avg=np.mean([c["methods"][f"mse_mlp_{seed}"][arm]["macro_rmse"] for seed in [2026111701,2026111702,2026111703]])
                    assert abs(avg-q[arm]["macro_rmse"])<1e-12
            data.append({"dataset":ds,"role":role,"method":key,"original_macro_rmse":a,"own_macro_rmse":b,"change_percent":v})
        y=np.arange(5);ax.barh(y,vals,color=cols,height=.60);ax.invert_yaxis();ax.set_yticks(y,labels,fontsize=8)
        ax.axvline(0,color="black",lw=.7);ax.set_xlim(-38,16);ax.set_xticks([-30,-20,-10,0,10])
        ax.grid(axis="x",alpha=.2,lw=.5);ax.set_axisbelow(True);ax.tick_params(axis="y",length=0,pad=4)
        ax.text(-.64,1.10,title,transform=ax.transAxes,ha="left",va="bottom",fontsize=8.5,fontweight="bold")
        for yy,v in zip(y,vals):
            if v < -25:
                ax.text(v+1.2,yy,f"{v:+.1f}",ha="left",va="center",fontsize=8,color="white")
            else:
                ax.text(v-.65 if v<0 else v+.65,yy,f"{v:+.1f}",ha="right" if v<0 else "left",va="center",fontsize=8)
    axs[-1].set_xlabel("Relative macro-RMSE change (%)",labelpad=3)
    fig.subplots_adjust(left=.40,right=.975,top=.91,bottom=.14,hspace=.55)
    fig.text(.5,.025,"Channel-wise vs ambient; negative = better.",ha="center",fontsize=8)
    NUMERIC["figure4"]=data;save(fig,"figure4-initialization-effect")
