from __future__ import annotations

import argparse
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "validation"
W0, T, MU, SIGMA, R = 100.0, 10.0, 0.0816, 0.1863, 0.00623
TARGETS = (125.0, 250.0)
MODELS = ("PCMV", "DOMV", "cTCMV", "dTCMV")
PUBLISHED = {
    125.0: {
        "PCMV": {"parameter":259.0,"mean":125.0,"median":127.0,"stdev":9.0,"skewness":-15.0,"excess_kurtosis":1042.0,"var_01":91.0,"var_05":113.0,"var_10":119.0,"cvar_01":63.0,"cvar_05":97.0,"cvar_10":107.0,"prob_below_risk_free":0.03,"prob_below_target":0.26,"conditional_mean_below_risk_free":87.0,"conditional_mean_below_target":117.0},
        "DOMV": {"parameter":0.111,"mean":125.0,"median":125.0,"stdev":16.0,"skewness":0.0,"excess_kurtosis":0.0,"var_01":88.0,"var_05":99.0,"var_10":105.0,"cvar_01":82.0,"cvar_05":92.0,"cvar_10":97.0,"prob_below_risk_free":0.12,"prob_below_target":0.50,"conditional_mean_below_risk_free":99.0,"conditional_mean_below_target":112.0},
        "cTCMV": {"parameter":0.044,"mean":125.0,"median":125.0,"stdev":15.0,"skewness":0.0,"excess_kurtosis":0.0,"var_01":91.0,"var_05":101.0,"var_10":106.0,"cvar_01":86.0,"cvar_05":95.0,"cvar_10":100.0,"prob_below_risk_free":0.10,"prob_below_target":0.50,"conditional_mean_below_risk_free":100.0,"conditional_mean_below_target":113.0},
        "dTCMV": {"parameter":0.041,"mean":125.0,"median":124.0,"stdev":16.0,"skewness":0.4,"excess_kurtosis":0.3,"var_01":92.0,"var_05":101.0,"var_10":105.0,"cvar_01":89.0,"cvar_05":96.0,"cvar_10":99.0,"prob_below_risk_free":0.11,"prob_below_target":0.53,"conditional_mean_below_risk_free":100.0,"conditional_mean_below_target":113.0},
    },
    250.0: {
        "PCMV": {"parameter":569.0,"mean":250.0,"median":269.0,"stdev":71.0,"skewness":-15.0,"excess_kurtosis":1042.0,"var_01":-15.0,"var_05":159.0,"var_10":206.0,"cvar_01":-228.0,"cvar_05":37.0,"cvar_10":112.0,"prob_below_risk_free":0.03,"prob_below_target":0.26,"conditional_mean_below_risk_free":-45.0,"conditional_mean_below_target":187.0},
        "DOMV": {"parameter":0.014,"mean":250.0,"median":250.0,"stdev":124.0,"skewness":0.0,"excess_kurtosis":0.0,"var_01":-38.0,"var_05":47.0,"var_10":92.0,"cvar_01":-80.0,"cvar_05":-5.0,"cvar_10":33.0,"prob_below_risk_free":0.12,"prob_below_target":0.50,"conditional_mean_below_risk_free":45.0,"conditional_mean_below_target":151.0},
        "cTCMV": {"parameter":0.006,"mean":250.0,"median":250.0,"stdev":112.0,"skewness":0.0,"excess_kurtosis":0.0,"var_01":-11.0,"var_05":65.0,"var_10":106.0,"cvar_01":-49.0,"cvar_05":19.0,"cvar_10":53.0,"prob_below_risk_free":0.10,"prob_below_target":0.50,"conditional_mean_below_risk_free":53.0,"conditional_mean_below_target":160.0},
        "dTCMV": {"parameter":0.001,"mean":250.0,"median":123.0,"stdev":444.0,"skewness":11.0,"excess_kurtosis":487.0,"var_01":8.0,"var_05":17.0,"var_10":27.0,"cvar_01":5.0,"cvar_05":11.0,"cvar_10":17.0,"prob_below_risk_free":0.45,"prob_below_target":0.72,"conditional_mean_below_risk_free":52.0,"conditional_mean_below_target":95.0},
    },
}
MC_METRICS = {"mean","median","stdev","var_01","var_05","var_10","cvar_01","cvar_05","cvar_10","prob_below_risk_free","prob_below_target","conditional_mean_below_risk_free","conditional_mean_below_target"}


def constants():
    a = ((MU - R) / SIGMA) ** 2
    return a, a * T, math.sqrt(a * T), W0 * math.exp(R * T)


def pcmv_exact(target):
    _, at, root_at, rf = constants()
    gamma = 2 * rf + 2 * math.exp(at) / math.expm1(at) * (target - rf)
    upper, scale, log_mu = gamma / 2, gamma / 2 - rf, -1.5 * at
    eln, ea, n = math.exp(-at), math.exp(at), NormalDist()
    out = {"parameter":gamma,"mean":upper-scale*eln,"median":upper-scale*math.exp(log_mu),"stdev":scale*math.exp(-at)*math.sqrt(math.expm1(at)),"skewness":-(ea+2)*math.sqrt(ea-1),"excess_kurtosis":math.exp(4*at)+2*math.exp(3*at)+3*math.exp(2*at)-6}
    for alpha in (0.01,0.05,0.10):
        z=n.inv_cdf(1-alpha); k=f"{int(alpha*100):02d}"
        out[f"var_{k}"]=upper-scale*math.exp(log_mu+root_at*z)
        out[f"cvar_{k}"]=upper-scale*eln*n.cdf(root_at-z)/alpha
    for name,threshold in (("risk_free",rf),("target",target)):
        z=(math.log((upper-threshold)/scale)-log_mu)/root_at; prob=1-n.cdf(z)
        out[f"prob_below_{name}"]=prob
        out[f"conditional_mean_below_{name}"]=upper-scale*eln*n.cdf(root_at-z)/prob
    return out


def normal_exact(target, model):
    _, at, _, rf = constants()
    if model == "DOMV":
        rho=math.expm1(at)/(2*(target-rf)); var=.5*(1/(2*rho))**2*math.expm1(2*at)
    else:
        rho=at/(2*(target-rf)); var=(1/(2*rho))**2*at
    sd, n = math.sqrt(var), NormalDist()
    out={"parameter":rho,"mean":target,"median":target,"stdev":sd,"skewness":0.0,"excess_kurtosis":0.0}
    for alpha in (0.01,0.05,0.10):
        z=n.inv_cdf(alpha); k=f"{int(alpha*100):02d}"
        out[f"var_{k}"]=target+sd*z
        out[f"cvar_{k}"]=target-sd*math.exp(-z*z/2)/(math.sqrt(2*math.pi)*alpha)
    for name,threshold in (("risk_free",rf),("target",target)):
        z=(threshold-target)/sd; prob=n.cdf(z)
        out[f"prob_below_{name}"]=prob
        out[f"conditional_mean_below_{name}"]=target-sd*math.exp(-z*z/2)/(math.sqrt(2*math.pi)*prob)
    return out


def dtcmv_exact(target):
    pub=PUBLISHED[target]["dTCMV"]; mean,sd=pub["mean"],pub["stdev"]
    lv=math.log1p((sd/mean)**2); ls=math.sqrt(lv); lm=math.log(mean)-lv/2; n=NormalDist(); ev=math.exp(lv)
    out={"parameter":pub["parameter"],"mean":mean,"median":math.exp(lm),"stdev":sd,"skewness":(ev+2)*math.sqrt(ev-1),"excess_kurtosis":math.exp(4*lv)+2*math.exp(3*lv)+3*math.exp(2*lv)-6,"_log_mean":lm,"_log_sd":ls}
    for alpha in (0.01,0.05,0.10):
        z=n.inv_cdf(alpha); k=f"{int(alpha*100):02d}"
        out[f"var_{k}"]=math.exp(lm+ls*z); out[f"cvar_{k}"]=mean*n.cdf(z-ls)/alpha
    _,_,_,rf=constants()
    for name,threshold in (("risk_free",rf),("target",target)):
        z=(math.log(threshold)-lm)/ls; prob=n.cdf(z)
        out[f"prob_below_{name}"]=prob; out[f"conditional_mean_below_{name}"]=mean*n.cdf(z-ls)/prob
    return out


def exact_statistics(target, model):
    return pcmv_exact(target) if model=="PCMV" else normal_exact(target,model) if model in {"DOMV","cTCMV"} else dtcmv_exact(target)


def sample_terminal(target, model, z):
    exact=exact_statistics(target,model)
    if model=="PCMV":
        _,at,root_at,rf=constants(); upper=exact["parameter"]/2
        return upper-(upper-rf)*np.exp(-1.5*at+root_at*z)
    if model in {"DOMV","cTCMV"}: return target+exact["stdev"]*z
    return np.exp(exact["_log_mean"]+exact["_log_sd"]*z)


def sample_statistics(values,target):
    _,_,_,rf=constants()
    out={"mean":float(values.mean()),"median":float(np.median(values)),"stdev":float(values.std()),"prob_below_risk_free":float(np.mean(values<=rf)),"prob_below_target":float(np.mean(values<=target)),"conditional_mean_below_risk_free":float(values[values<=rf].mean()),"conditional_mean_below_target":float(values[values<=target].mean())}
    for alpha in (0.01,0.05,0.10):
        k=f"{int(alpha*100):02d}"; q=float(np.quantile(values,alpha)); out[f"var_{k}"]=q; out[f"cvar_{k}"]=float(values[values<=q].mean())
    return out


def is_anchor(model,metric): return model=="dTCMV" and metric in {"parameter","mean","stdev"}

def published_tolerance(model,metric,published):
    if metric.startswith("prob_"): return .011
    if metric=="parameter": return 1.0 if model=="PCMV" else .0006
    if metric in {"skewness","excess_kurtosis"} and abs(published)<1: return .06
    return 1.0

def mc_tolerance(metric,exact,model):
    if metric.startswith("prob_"): return .00075 if model=="dTCMV" else .0005
    if metric in {"mean","median","stdev"}: return max(.08,(.008 if model=="dTCMV" else .004)*abs(exact))
    if metric in {"var_01","cvar_01","conditional_mean_below_risk_free"}: return max(.9,.015*abs(exact))
    return max(.45,.008*abs(exact))


def build_tables(samples,seed):
    a,at,_,rf=constants()
    parameters=pd.DataFrame([{"parameter":k,"published":p,"implementation":v} for k,p,v in [("w0",W0,W0),("T",T,T),("mu",MU,MU),("sigma",SIGMA,SIGMA),("r",R,R),("A=((mu-r)/sigma)^2",a,a),("A*T",at,at),("w0*exp(rT)",106.43,rf),("contribution",0.0,0.0),("leverage_constraint",0.0,0.0),("transaction_cost",0.0,0.0)]])
    parameters["absolute_difference"]=abs(parameters["implementation"]-parameters["published"])
    z=np.random.default_rng(seed).standard_normal(samples); rows=[]
    for target in TARGETS:
        for model in MODELS:
            exact=exact_statistics(target,model); mc=sample_statistics(sample_terminal(target,model,z),target)
            for metric,published in PUBLISHED[target][model].items():
                anchor=is_anchor(model,metric); ev=exact[metric]; ed=abs(ev-published); ep=True if anchor else ed<=published_tolerance(model,metric,published)
                mv=mc.get(metric,math.nan); md=abs(mv-ev) if math.isfinite(mv) else math.nan; mp=md<=mc_tolerance(metric,ev,model) if metric in MC_METRICS else True
                method="reflected_lognormal_closed_form" if model=="PCMV" else "normal_closed_form" if model in {"DOMV","cTCMV"} else "lognormal_distribution_reconstruction"
                rows.append({"target_mean":target,"model":model,"metric":metric,"published_table_5_1":published,"exact_or_reconstructed":ev,"monte_carlo":mv,"abs_exact_minus_published":ed,"abs_mc_minus_exact":md,"anchor_metric":anchor,"published_rounding_check":ep,"monte_carlo_check":mp,"overall_check":ep and mp,"reproduction_method":method,"samples":samples,"seed":seed})
    detail=pd.DataFrame(rows); summaries=[]
    for target in TARGETS:
        for model in MODELS:
            part=detail[(detail.target_mean==target)&(detail.model==model)]; testable=part[~part.anchor_metric]; mcpart=part[part.metric.isin(MC_METRICS)]
            summaries.append({"target_mean":target,"model":model,"method":part.reproduction_method.iloc[0],"testable_published_metrics":len(testable),"published_metrics_passed":int(testable.published_rounding_check.sum()),"mc_metrics_tested":len(mcpart),"mc_metrics_passed":int(mcpart.monte_carlo_check.sum()),"overall_pass":bool(part.overall_check.all()),"max_abs_exact_minus_published_nonanchor":float(testable.abs_exact_minus_published.max())})
    return parameters,detail,pd.DataFrame(summaries)


def main():
    parser=argparse.ArgumentParser(description="External validation of four MV concepts against van Staden et al. (2021), Table 5.1")
    parser.add_argument("--samples",type=int,default=2_000_000); parser.add_argument("--seed",type=int,default=20_210_419); args=parser.parse_args()
    if args.samples<100_000: raise ValueError("use at least 100,000 paths for tail validation")
    parameters,detail,summary=build_tables(args.samples,args.seed); RESULTS.mkdir(parents=True,exist_ok=True)
    parameters.to_csv(RESULTS/"vanstaden2021_all_mv_parameter_match.csv",index=False); detail.to_csv(RESULTS/"vanstaden2021_all_mv_reproduction.csv",index=False); summary.to_csv(RESULTS/"vanstaden2021_all_mv_external_summary.csv",index=False)
    print(summary.to_string(index=False))
    if not bool(summary.overall_pass.all()): raise RuntimeError("at least one external-validation check failed")

if __name__=="__main__": main()
