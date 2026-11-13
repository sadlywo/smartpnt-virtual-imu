"""ALiMU evidence figures. All drawing and exports use Matplotlib only."""
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Editable text is mandatory; use the repository's established sans-serif font.
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'

from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from PIL import Image

from .style import apply_paper_style

LABELS = {
    "cnn_lstm64": "CNN–LSTM64", "cnn_bilstm32": "CNN–BiLSTM32",
    "cnn_identity": "CNN only", "cnn_gru": "CNN–GRU", "cnn_bigru": "CNN–BiGRU",
    "cnn_transformer": "CNN–Transformer", "cnn_tcn": "CNN–TCN",
    "linear_lstm64": "Linear–LSTM64", "mlp_lstm64": "MLP–LSTM64",
    "gru_lstm64": "GRU–LSTM64", "transformer_lstm64": "Transformer–LSTM64",
    "raw": "Raw LOW", "affine": "Affine",
    "ddpm": "DDPM–Transformer", "cnn_bilstm": "CNN–BiLSTM",
    "informer": "Informer", "transformer": "Transformer",
    "no_diffusion": "No diffusion (tuned)",
    "no_diffusion_matched": "No diffusion (strict)",
}
BLUE, TEAL, BLACK = "#0F4D92", "#42949E", "#272727"
LILAC, VIOLET, RAW = "#7884B4", "#9A4D8E", "#C95F3F"
SEEDS = (20261010, 20261011, 20261012)

def color(v):
    if v in ("cnn_gru", "informer"): return BLUE
    if v in ("cnn_bigru", "no_diffusion"): return TEAL
    if v in ("cnn_lstm64", "no_diffusion_matched"): return BLACK
    if v in ("raw", "ddpm"): return RAW
    if v.endswith("lstm64") and not v.startswith("cnn"): return VIOLET
    return LILAC

def marker(v):
    return {"cnn_gru": "o", "cnn_bigru": "D", "cnn_lstm64": "s"}.get(v, "o")

def style():
    apply_paper_style()
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
        "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.size": 8, "axes.titlesize": 8.3, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": .65, "axes.grid": False, "legend.frameon": False, "savefig.facecolor": "white"})

def panel(ax, letter, title):
    ax.set_title(title, loc="left", pad=12, fontweight="normal")
    ax.text(-.055, 1.045, letter, transform=ax.transAxes, weight="bold", fontsize=9,
            va="bottom", ha="right")
    ax.grid(axis="x", color="#E5E7EB", linewidth=.5)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=.6)

def rows(ax, variants, labels=True):
    ax.set_yticks(range(len(variants)), [LABELS[v] for v in variants] if labels else [])
    ax.set_ylim(len(variants)-.45, -.55)
    ax.tick_params(axis="y", length=0, pad=6)
    if labels:
        for t,v in zip(ax.get_yticklabels(),variants):
            if v in ("cnn_gru","cnn_bigru","cnn_lstm64"):
                t.set_color(color(v)); t.set_fontweight("bold")

def point(ax, values, y, v, scale=1., offset=0.):
    a=np.asarray(values,dtype=float)*scale
    if not len(a) or not np.isfinite(a).all(): raise ValueError("Missing/nonfinite plot data")
    c=color(v); center=a.mean(); sd=a.std(ddof=1) if len(a)>1 else 0.
    ax.errorbar(center,y+offset,xerr=sd,fmt=marker(v),ms=4.2,mfc=c,mec="white",mew=.45,
                color=c,elinewidth=1.1,capsize=2,capthick=.8,zorder=4)
    if len(a)>1:
        for x,j in zip(a,np.linspace(-.075,.075,len(a))):
            ax.plot(x,y+offset+j,"o",ms=2.3,mfc="white",mec=c,mew=.65,zorder=5)
    return center,sd

def header(fig, text, subtitle=None):
    fig.text(.5,.972,text,ha="center",va="top",fontsize=10,weight="bold")
    if subtitle: fig.text(.5,.932,subtitle,ha="center",va="top",fontsize=7.2,color="#555555")

def foot(fig,text,y=.045):
    fig.text(.5,y,text,ha="center",va="bottom",fontsize=6.7,color="#555555",linespacing=1.45)

def highlights(fig,y=.13):
    h=[Line2D([],[],color=color(v),marker=marker(v),lw=0,label=l) for v,l in [
        ("cnn_gru","VALID-selected: CNN–GRU"),("cnn_bigru","Observed TEST minimum: CNN–BiGRU"),
        ("cnn_lstm64","Reference: CNN–LSTM64")]]
    fig.legend(handles=h,loc="center",bbox_to_anchor=(.5,y),ncol=2,frameon=False,columnspacing=1.8,handletextpad=.3)

def _values(frame, variant, **selectors):
    mask=frame.variant.eq(variant)
    for key,value in selectors.items(): mask &= frame[key].eq(value)
    out=frame.loc[mask].sort_values("seed")
    if out.empty: raise ValueError("Missing plot cell: "+str((variant,selectors)))
    return out

def selection(data):
    d=data["shared_seed"]
    order=data["order"]
    fig,axs=plt.subplots(1,2,figsize=(7.2,4.7))
    fig.subplots_adjust(left=.245,right=.975,bottom=.29,top=.84,wspace=.18)
    header(fig,"ALiMU | selection and observed test performance", "One shared model for three LOW devices → VN100 · 320-sample context · lower is better")
    for ax,split,letter,title in zip(axs,["validation","test"],"ab",["Validation · selects the model","Test · describes held-out error"]):
        for i,v in enumerate(order):
            vals=_values(d,v,protocol="alimu_only",split=split,aggregation="dataset",dataset_id="alimu").normalized_score
            m,_=point(ax,vals,i,v)
            if v in ("cnn_gru","cnn_bigru","cnn_lstm64"):
                ax.text(.805,i,f"{m:.4f}",ha="right",va="center",fontsize=6.8,color=color(v))
        panel(ax,letter,title); rows(ax,order,ax is axs[0]);ax.set_xlim(.25,.82)
        ax.set_xticks([.3,.4,.5,.6,.7,.8]);ax.set_xlabel("Balanced normalized RMSE score")
    highlights(fig,.148)
    foot(fig,"Mean ± sample SD; small points: three optimization seeds. Raw score = 1 (outside displayed range).\nValidation and test use their own Raw denominators; test was previously inspected. Two physical test recordings.")
    return fig

def physical(data):
    d=data["shared_seed"];order=["raw"]+data["order"]
    fig,axs=plt.subplots(1,2,figsize=(7.2,4.8));fig.subplots_adjust(left=.245,right=.975,bottom=.29,top=.84,wspace=.18)
    header(fig,"ALiMU | physical signal errors", "Shared ALiMU-only training · component RMSE against measured VN100 · lower is better")
    for ax,metric,scale,letter,title,xlab in zip(axs,["gyro_rmse","accel_rmse"],[1000,1],"ab",["Gyroscope","Accelerometer"],["RMSE (10⁻³ rad s⁻¹)","RMSE (m s⁻²)"]):
        for i,v in enumerate(order):
            vals=_values(d,v,protocol="alimu_only",split="test",aggregation="dataset",dataset_id="alimu")[metric]
            m,_=point(ax,vals,i,v,scale)
            if v in ("raw","cnn_gru","cnn_bigru","cnn_lstm64"):
                tx=4.38 if scale==1000 else .735
                ax.text(tx,i,f"{m:.3f}",ha="right",va="center",fontsize=6.8,color=color(v))
        panel(ax,letter,title);rows(ax,order,ax is axs[0]);ax.set_xlabel(xlab);ax.set_xlim(0,4.5 if scale==1000 else .76)
    highlights(fig,.148)
    foot(fig,"Per seed: pool sample squared errors across all three devices, then take the square root.\nMean ± sample SD across three seeds; Raw is deterministic (one value). Each seed: 88,080 device–time target rows.")
    return fig

def ablations(data):
    d=data["shared_seed"];p=data["paired_seed"];w=data["window_seed"]
    fig,axs=plt.subplots(2,2,figsize=(7.2,6.8));fig.subplots_adjust(left=.235,right=.98,bottom=.13,top=.85,hspace=.58,wspace=.55)
    header(fig,"ALiMU | what the ablations support", "Shared training · fixed recipe · directional effects without significance claims")
    back=["cnn_bilstm32","cnn_identity","cnn_gru","cnn_bigru","cnn_transformer","cnn_tcn"]
    front=["linear_lstm64","mlp_lstm64","gru_lstm64","transformer_lstm64"]
    for ax,variants,letter,title in [(axs[0,0],back,"a","Replace the recurrent backend"),(axs[0,1],front,"b","Replace the CNN frontend")]:
        for i,v in enumerate(variants):
            vals=_values(p,v,protocol="alimu_only",split="test",aggregation="dataset",dataset_id="alimu",metric="normalized_score").relative_delta_pct
            point(ax,vals,i,v)
        panel(ax,letter,title);rows(ax,variants);ax.axvline(0,color=BLACK,lw=.8,ls="--")
        ax.set_xlabel("Same-seed score change (%)")
        if letter=="a":
            ax.set_xlim(-12,30);ax.set_xticks([-10,0,10,20,30])
        else:
            ax.set_xlim(0,105);ax.set_xticks([0,25,50,75,100])
    ax=axs[1,0]
    for split,c,m in [("validation",BLUE,"o"),("test",BLACK,"s")]:
        means=[];sds=[]
        for win in [40,80,160,320]:
            v=w[(w.protocol=="alimu_only")&(w.split==split)&(w.variant=="cnn_bilstm")&(w.window==win)&(w.aggregation=="dataset")&(w.dataset_id=="alimu")].normalized_score.to_numpy()
            means.append(v.mean());sds.append(v.std(ddof=1))
        ax.errorbar([1,2,4,8],means,yerr=sds,color=c,marker=m,ms=3.5,lw=1,capsize=2,label=split.upper())
    panel(ax,"c","CNN–BiLSTM window sensitivity");ax.set_xlabel("Input context at 40 Hz (s)");ax.set_ylabel("Normalized score")
    ax.set_xticks([1,2,4,8]);ax.set_ylim(.32,.415);ax.legend(loc="upper right",fontsize=6.5)
    ax=axs[1,1];pairs=["BMX160_to_VN100","LSM9DS1_to_VN100","MPU9250_to_VN100"]
    for v,dy in [("cnn_gru",-.17),("cnn_bigru",0),("cnn_lstm64",.17)]:
        for i,pair in enumerate(pairs):
            vals=_values(d,v,protocol="alimu_only",split="test",aggregation="pair",dataset_id="alimu",pair_id=pair).normalized_score
            point(ax,vals,i,v,offset=dy)
    panel(ax,"d","Device-specific test scores");ax.set_yticks(range(3),["BMX160","LSM9DS1","MPU9250"]);ax.set_ylim(2.55,-.55)
    ax.set_xlabel("Normalized score");ax.set_xlim(.24,.55)
    highlights(fig,.495)
    foot(fig,"a,b: negative changes favor the replacement; reference is CNN–LSTM64. Points and bars: three seeds, mean ± SD.\nc: earlier frozen window study; VALID favors 4 s, observed TEST favors 8 s. d: all devices use the same shared model.\nSeven new non-deletion replacements are parameter matched within ±5%; BiLSTM32 and deletion controls are outside this set.",y=.025)
    return fig

def matched(data):
    d=data["matched_seed"];p=data["strict_seed"]
    order=["raw","affine","ddpm","cnn_bilstm","informer","transformer","no_diffusion","no_diffusion_matched"]
    fig,axs=plt.subplots(2,2,figsize=(7.2,6.2));fig.subplots_adjust(left=.25,right=.98,bottom=.16,top=.86,hspace=.75,wspace=.30)
    header(fig,"ALiMU | a separate per-device model comparison", "80-sample context · three independently trained LOW→VN100 mappings · lower RMSE is better")
    for ax,metric,scale,l,title,xlab in zip(axs[0],["gyro","accel"],[1000,1],"ab",["Gyro · Informer lowest","Accel · no diffusion lowest"],["RMSE (10⁻³ rad s⁻¹)","RMSE (m s⁻²)"]):
        for i,v in enumerate(order):
            vals=d[(d.pair_id=="__all_pairs__")&(d.component==metric)&(d.method==v)].sort_values("seed").rmse
            point(ax,vals,i,v,scale)
        panel(ax,l,title);rows(ax,order,ax is axs[0,0]);ax.set_xlim(left=0);ax.set_xlabel(xlab)
    pairs=["BMX160_to_VN100","LSM9DS1_to_VN100","MPU9250_to_VN100"]
    for ax,metric,l,title in zip(axs[1],["gyro","accel"],"cd",["Strict diffusion ablation · gyro","Strict diffusion ablation · accel"]):
        for i,pair in enumerate(pairs):
            a=p[(p.pair_id==pair)&(p.component==metric)].sort_values("seed")
            point(ax,a.ddpm_change_pct,i,"ddpm")
        panel(ax,l,title);ax.set_yticks(range(3),[x.split("_to_")[0] for x in pairs] if ax is axs[1,0] else [])
        ax.set_ylim(2.5,-.5);ax.axvline(0,color=BLACK,ls="--",lw=.8);ax.set_xlim(-5,140);ax.set_xticks([0,40,80,120]);ax.set_xlabel("DDPM error change vs strict removal (%)")
    foot(fig,"a,b: means ± SD of three seeds; Raw/Affine are deterministic. Pooled per seed: 89,949 device–time rows.\nc,d: same-seed, same selected backbone/configuration/initialization. Every one of the 18 deltas is positive.\nThis 80-sample protocol differs from the shared 320-sample study; absolute errors must not be ranked across the two.")
    return fig

def scope_cost(data):
    d=data["shared_seed"];c=data["compute_seed"];order=data["order"]
    fig,axs=plt.subplots(1,2,figsize=(7.2,4.9));fig.subplots_adjust(left=.245,right=.975,bottom=.29,top=.84,wspace=.23)
    header(fig,"ALiMU | training scope and inference cost", "All test points below evaluate ALiMU · one shared model per training scope")
    ax=axs[0]
    for i,v in enumerate(order):
        for protocol,offset,m in [("alimu_only",-.13,"o"),("pooled",.13,"s")]:
            vals=_values(d,v,protocol=protocol,split="test",aggregation="dataset",dataset_id="alimu").normalized_score
            a=vals.to_numpy();cc=color(v)
            ax.errorbar(a.mean(),i+offset,xerr=a.std(ddof=1),fmt=m,ms=3.5,color=cc,mfc=cc if protocol=="alimu_only" else "white",capsize=1.5,lw=.8)
    panel(ax,"a","Test error under two training scopes");rows(ax,order);ax.set_xlim(.30,.76);ax.set_xlabel("Normalized score on ALiMU")
    fig.legend(handles=[Line2D([],[],color=BLACK,marker="o",lw=0,label="ALiMU-only training"),Line2D([],[],color=BLACK,marker="s",mfc="white",lw=0,label="ALiMU + Penn training")],loc="center",bbox_to_anchor=(.5,.16),ncol=2,fontsize=7)
    ax=axs[1]
    for i,v in enumerate(order):
        a=c[(c.protocol=="alimu_only")&(c.variant==v)].sort_values("seed").batch1_ms_mean
        point(ax,a,i,v)
    panel(ax,"b","Uniform batch-1 CUDA latency");rows(ax,order,False);ax.set_xlim(0,3.25);ax.set_xlabel("Forward latency (ms)")
    foot(fig,"Mean ± sample SD across three optimization seeds. Joint-scope VALID selects CNN–LSTM64 using both datasets.\nRTX 3060 Ti; synchronized timing, 5 warmups and 20 repeats per fit. Forward time excludes buffering/future-context wait.\nThe fixed training recipe has no per-family tuning; the results identify candidates under these specific conditions.")
    return fig

BUILDERS=[("fig1_shared_selection",selection), ("fig2_shared_physical",physical),
          ("fig3_ablation_evidence",ablations), ("fig4_per_device_ddpm",matched),
          ("fig5_scope_and_cost",scope_cost)]

def render_figures(data, output, config):
    """Export all formats; reject out-of-canvas labels and non-editable SVG text."""
    style();output=Path(output);output.mkdir(parents=True,exist_ok=True);checks=[]
    for stem,builder in BUILDERS:
        fig=builder(data)
        fig.canvas.draw();renderer=fig.canvas.get_renderer();outside=[]
        # Matplotlib may instantiate auto ticks outside the view; these are not drawn.
        undrawn_tick_ids=set()
        for ax in fig.axes:
            for tick in ax.xaxis.get_major_ticks():
                if not min(ax.get_xlim()) <= tick.get_loc() <= max(ax.get_xlim()):
                    undrawn_tick_ids.update([id(tick.label1),id(tick.label2)])
            for tick in ax.yaxis.get_major_ticks():
                if not min(ax.get_ylim()) <= tick.get_loc() <= max(ax.get_ylim()):
                    undrawn_tick_ids.update([id(tick.label1),id(tick.label2)])
        # Reject clipped visible labels before any export; no bbox-tight resizing.
        for text in fig.findobj(matplotlib.text.Text):
            if id(text) in undrawn_tick_ids or not text.get_visible() or not text.get_text().strip():continue
            b=text.get_window_extent(renderer)
            if b.x0 < -1 or b.y0 < -1 or b.x1 > fig.bbox.width+1 or b.y1 > fig.bbox.height+1:
                outside.append(text.get_text())
        if outside: raise ValueError(stem+" out-of-canvas text: "+repr(outside))
        for ext in config['render']['formats']:
            path=output/(stem+'.'+ext)
            opts={"dpi": config['render']['tiff_dpi'] if ext=='tiff' else config['render']['png_dpi']}
            if ext=='tiff':opts['pil_kwargs']={"compression":"tiff_lzw"}
            fig.savefig(path,**opts)
        root=ET.parse(output/(stem+'.svg')).getroot()
        text_count=sum(1 for el in root.iter() if el.tag.endswith('}text'))
        if text_count<15:raise ValueError("SVG editable text missing")
        with Image.open(output/(stem+'.png')) as im:
            png_size=list(im.size);png_dpi=list(im.info.get('dpi',()))
        with Image.open(output/(stem+'.tiff')) as im:
            tiff_size=list(im.size);tiff_dpi=[float(x) for x in im.info.get('dpi',())]
        checks.append(dict(figure=stem,width_inches=fig.get_size_inches()[0],height_inches=fig.get_size_inches()[1],
            svg_text_elements=text_count,png_pixels=png_size,png_dpi=png_dpi,tiff_pixels=tiff_size,tiff_dpi=tiff_dpi,
            clipped_text=outside,status='PASS'))
        plt.close(fig)
    return checks
