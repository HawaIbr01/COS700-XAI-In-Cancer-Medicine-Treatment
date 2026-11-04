"""Final-run figures for COS700 (per-drug rule extraction).
Usage (from the repo root, e.g. in the VS Code terminal):
    python scripts/make_final_figures.py --in-dir artifacts/per_drug/final_results --out-dir artifacts/per_drug/final_results/figures
Reads results_drug_level.csv (one row per drug x arm x extractor). Writes fig0-fig3 PNGs.
Needs pandas, numpy, matplotlib only.
"""
import argparse, os
ap=argparse.ArgumentParser(); ap.add_argument("--in-dir",required=True); ap.add_argument("--out-dir",required=True)
A=ap.parse_args(); os.makedirs(A.out_dir,exist_ok=True)
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":150})
d=pd.read_csv(os.path.join(A.in_dir,"results_drug_level.csv"))
C={"adam":"#7a8691","adam_wd":"#E69F00","sam":"#0072B2"}
L={"adam":"Adam","adam_wd":"Adam+WD","sam":"SAM+WD"}
EX={"ga":"GA","nsga":"NSGA-II"}
def wide(m,e): return d[d.extractor==e].pivot(index="drug",columns="arm",values=m)
comps=[("sam","adam_wd","SAM+WD − Adam+WD\n(clean SAM effect)"),("adam_wd","adam","Adam+WD − Adam\n(weight decay + stopping rule)"),("sam","adam","SAM+WD − Adam\n(proposal contrast, confounded)")]
# Fig1 paired differences, fidelity
fig,ax=plt.subplots(2,3,figsize=(11,6),sharex=True)
for i,e in enumerate(EX):
    w=wide("heldout_fidelity",e)
    for j,(a,b,t) in enumerate(comps):
        x=(w[a]-w[b]).dropna(); ax[i,j].hist(x,bins=np.linspace(-0.4,0.4,41),color="#56B4E9",edgecolor="white")
        ax[i,j].axvline(0,color="k",lw=1); ax[i,j].axvline(x.median(),color="#D55E00",lw=2)
        ax[i,j].set_title(f"{EX[e]}: {t}" if i==0 else EX[e],fontsize=9)
        ax[i,j].text(0.97,0.92,f"median {x.median():+.3f}",transform=ax[i,j].transAxes,ha="right",color="#D55E00",fontsize=9)
        if j==0: ax[i,j].set_ylabel("drugs")
for a in ax[1]: a.set_xlabel("per-drug difference in held-out fidelity (macro-F1)")
fig.suptitle("Per-drug paired differences in fidelity (n = 379 drugs); orange = median",fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(A.out_dir,"fig1_fidelity_paired_differences.png")); plt.close()
# Fig2 scatter SAM+WD vs Adam+WD
fig,ax=plt.subplots(1,2,figsize=(9,4.5))
for k,e in enumerate(EX):
    w=wide("heldout_fidelity",e); ax[k].scatter(w.adam_wd,w.sam,s=8,alpha=.5,color=C["sam"],edgecolor="none")
    ax[k].plot([0,1],[0,1],"k",lw=1); ax[k].set_xlim(0,1);ax[k].set_ylim(0,1);ax[k].set_aspect("equal")
    ax[k].set_title(EX[e]); ax[k].set_xlabel("Adam+WD fidelity"); ax[k].set_ylabel("SAM+WD fidelity")
fig.suptitle("Fidelity per drug: points on the diagonal = no difference",fontsize=11); fig.tight_layout(); fig.savefig(os.path.join(A.out_dir,"fig2_fidelity_scatter_sam_vs_adamwd.png")); plt.close()
# Fig3 arm means (median across drugs of drug values) four metrics
ms=[("heldout_fidelity","Held-out fidelity"),("phi","PROMETHEE II φ"),("length","Rule length (conditions)"),("stability_jaccard_folds","Fold Jaccard stability")]
fig,ax=plt.subplots(1,4,figsize=(13,3.8))
for k,(m,t) in enumerate(ms):
    for ei,e in enumerate(EX):
        for ai,a in enumerate(L):
            v=d[(d.extractor==e)&(d.arm==a)][m].dropna()
            ax[k].scatter(ei+(ai-1)*0.2,v.mean(),s=70,color=C[a],label=L[a] if (k==0 and ei==0) else None,zorder=3)
            ax[k].plot([ei+(ai-1)*0.2]*2,[v.quantile(.25),v.quantile(.75)],color=C[a],lw=2)
    ax[k].set_xticks([0,1]);ax[k].set_xticklabels(["GA","NSGA-II"]);ax[k].set_title(t,fontsize=10)
ax[0].legend(frameon=False,fontsize=8)
fig.suptitle("Mean over drugs (dot) with interquartile range of drugs (bar). GA and NSGA-II not compared with each other",fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(A.out_dir,"fig3_arm_summaries.png")); plt.close()
# Fig4 pipeline
fig,ax=plt.subplots(figsize=(15,3)); ax.axis("off")
steps=["GDSC drug\n(379 drugs)","Network class\n(Adam / Adam+WD /\nSAM+WD)","5 folds by\ncell line","GA & NSGA-II\n10 seeds, 50 gen\nper fold","Pool of\nnon-dominated\nrules","Score on\nheld-out fold:\nfidelity, length,\nφ, stability","Paired Wilcoxon\nacross drugs\n+ BH"]
for i,s in enumerate(steps):
    ax.text(i/(len(steps)-1)*0.93+0.04,0.5,s,ha="center",va="center",fontsize=8,bbox=dict(boxstyle="round,pad=0.4",fc="#EAF2FA",ec="#0072B2"),transform=ax.transAxes)
    if i<len(steps)-1: ax.annotate("",xy=((i+1)/(len(steps)-1)*0.93+0.04-0.075,0.5),xytext=(i/(len(steps)-1)*0.93+0.04+0.075,0.5),xycoords="axes fraction",arrowprops=dict(arrowstyle="->"))
fig.savefig(os.path.join(A.out_dir,"fig0_pipeline.png"),bbox_inches="tight"); plt.close()