#!/usr/bin/env python3
"""Reproduce revision analyses from anonymised inputs; see README for provenance."""
from pathlib import Path
import sys,itertools,json
import numpy as np,pandas as pd
from scipy import stats,signal
from scipy.special import expit
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from patsy import build_design_matrices
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'original'));import analysis as a
OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
fb,srm=a.load_data();cell,esrm=a.srm_participant_cells(srm)
fbcell=a.participant_means(fb,['session','hrtf','location_deg'])
fba=a.participant_means(fb,['session','hrtf'])
fw=fba.pivot(index='participant',columns=['session','hrtf'],values='accuracy_pct')
def write(d,name):d.to_csv(OUT/(name+'.csv'),index=False);return d
def contrast(x,name):
 x=np.array(x,float);t=stats.ttest_1samp(x,0);ci=stats.t.interval(.95,len(x)-1,loc=x.mean(),scale=stats.sem(x))
 return dict(contrast=name,n=len(x),mean_pp=x.mean(),t=t.statistic,p=t.pvalue,ci_low=ci[0],ci_high=ci[1])
# H1 omitted from standalone summary, independently reconstructed here.
w=esrm[esrm.configuration.isin(a.SEPARATED)].groupby(['participant','session','hrtf']).srm_pp.mean().unstack(['session','hrtf'])
write(pd.DataFrame([contrast(w['before','optimised']-w['before','individual'],'H1')]),'h1')
# Marginal accuracy, front response probability, sensitivity and criterion.
rows=[]
for (pid,ses,h),g in fb.groupby(['participant','session','hrtf']):
 f=g[g.location_deg==30];b=g[g.location_deg==150]
 H=(f.correct.sum()+.5)/(len(f)+1);F=((1-b.correct).sum()+.5)/(len(b)+1)
 rows.append(dict(participant=pid,session=ses,hrtf=h,accuracy_pct=100*g.correct.mean(),front_response_pct=100*(f.correct.sum()+(1-b.correct).sum())/len(g),dprime=stats.norm.ppf(H)-stats.norm.ppf(F),criterion=-.5*(stats.norm.ppf(H)+stats.norm.ppf(F))))
sdt=write(pd.DataFrame(rows),'front_back_sensitivity_bias')
write(sdt.groupby(['session','hrtf']).agg(accuracy_pct=('accuracy_pct','mean'),front_response_pct=('front_response_pct','mean'),dprime=('dprime','mean'),criterion=('criterion','mean')).reset_index(),'front_back_summary')
# Binomial ceiling sensitivity, saturated cell model; cluster bias-reduced covariance.
fb['post']=(fb.session=='after').astype(int);fb['opt']=(fb.hrtf=='optimised').astype(int);fb['rear']=(fb.location_deg==150).astype(int)
fit=smf.gee('correct ~ post * opt * rear',groups='participant',data=fb,family=sm.families.Binomial(),cov_struct=sm.cov_struct.Independence()).fit(cov_type='bias_reduced')
new=pd.DataFrame(list(itertools.product([0,1],repeat=3)),columns=['post','opt','rear'])
X=np.asarray(build_design_matrices([fit.model.data.design_info],new)[0]);p=expit(X@fit.params)
c=np.where(new.post==1,1,-1)*np.where(new.opt==1,1,-1)/2
est=float(c@p);grad=(c*p*(1-p))@X
se=float(np.sqrt(grad@fit.cov_params()@grad));t=est/se;pval=2*stats.t.sf(abs(t),17);q=stats.t.ppf(.975,17)
write(pd.DataFrame([dict(contrast='H3 probability-scale difference of changes',estimate_pp=100*est,se_pp=100*se,t=t,p=pval,ci_low_pp=100*(est-q*se),ci_high_pp=100*(est+q*se),clusters=18,covariance='Mancl-DeRouen bias-reduced; t17 reference')]),'front_back_binomial_sensitivity')
# Exact sign-flip test on H3 participant contrasts (symmetric-null sensitivity).
d=(fw['after','optimised']-fw['before','optimised'])-(fw['after','individual']-fw['before','individual'])
sgn=2*((np.arange(2**18)[:,None]>>np.arange(18))&1)-1
perm=np.mean(abs(sgn@d.to_numpy()/18)>=abs(d.mean())-1e-10)
write(pd.DataFrame([dict(contrast='H3 exact two-sided sign flip',p=perm,permutations=2**18)]),'front_back_permutation')
# Raw accuracy and within-HRTF/session baseline subtraction decomposition.
write(cell.groupby(['session','hrtf','configuration']).accuracy_pct.agg(['mean','sem']).reset_index(),'speech_accuracy')
write(esrm[esrm.configuration.isin(a.SEPARATED)].groupby(['session','hrtf','configuration']).srm_pp.agg(['mean','sem']).reset_index(),'behavioural_srm')
cw=cell.pivot(index='participant',columns=['session','hrtf','configuration'],values='accuracy_pct')
rows=[]
for h in a.HRTFS:
 for config in ['T30 M30',*a.SEPARATED]:
  delta=cw['after',h,config]-cw['before',h,config]
  rows.append(contrast(delta,f'{h}: {config}, post - pre'))
write(pd.DataFrame(rows),'speech_changes')
# Canonical predictor: full 20-mixture means, independent of missing behavioural rows.
cp=pd.read_csv(ROOT/'inputs/canonical_predictions.csv')
cm=cp.groupby(['participant','hrtf','configuration']).predicted_better_ear_srm_dB.mean().rename('effective_snr_db').reset_index()
co=cm[cm.configuration=='T30 M30'].drop(columns='configuration').rename(columns={'effective_snr_db':'colocated_effective_snr_db'})
cm=cm.merge(co,on=['participant','hrtf']);cm['predicted_release_db']=cm.effective_snr_db-cm.colocated_effective_snr_db
write(cm,'canonical_model_means')
points=esrm[esrm.configuration.isin(a.SEPARATED)].merge(cm,on=['participant','hrtf','configuration'],validate='many_to_one')
write(points,'prediction_points')
rows=[]
for ses in a.SESSIONS:
 for cfg in a.SEPARATED:
  g=points[(points.session==ses)&(points.configuration==cfg)&(points.hrtf=='individual')]
  res=stats.pearsonr(g.predicted_release_db,g.srm_pp);ci=res.confidence_interval()
  rows.append(dict(session=ses,configuration=cfg,n=len(g),r=res.statistic,p=res.pvalue,ci_low=ci.low,ci_high=ci.high))
r=pd.DataFrame(rows);r['p_holm']=multipletests(r.p,method='holm')[1];write(r,'within_configuration_correlations')
# Prediction GEE adjusted for all three configurations (supplementary only).
srm['post']=(srm.session=='after').astype(int)
i=srm[srm.hrtf=='individual'].copy();i['predictor_within']=i.predicted_srm_db-i.groupby('configuration').predicted_srm_db.transform('mean');i['predictor_within']/=i.predictor_within.std(ddof=1)
f=smf.gee('correct ~ predictor_within * post + C(configuration) * post',groups='participant',data=i,family=sm.families.Binomial(),cov_struct=sm.cov_struct.Exchangeable()).fit(cov_type='bias_reduced')
rows=[]
for name in f.params.index:
 b=f.params[name];se=f.bse[name];crit=stats.t.ppf(.975,17)
 rows.append(dict(term=name,estimate=b,se=se,OR=np.exp(b),OR_ci_low=np.exp(b-crit*se),OR_ci_high=np.exp(b+crit*se),p=2*stats.t.sf(abs(b/se),17)))
write(pd.DataFrame(rows),'configuration_adjusted_gee')
# Pre/post associations avoid mathematical coupling; do not imply treatment response.
rows=[]
sep=cell[cell.configuration.isin(a.SEPARATED)].groupby(['participant','session','hrtf']).accuracy_pct.mean().unstack(['session','hrtf'])
for outcome,wide in [('Front-back accuracy',fw),('Separated speech accuracy',sep)]:
 for h in a.HRTFS:
  res=stats.pearsonr(wide['before',h],wide['after',h]);ci=res.confidence_interval()
  rows.append(dict(outcome=outcome,hrtf=h,n=18,r=res.statistic,p=res.pvalue,ci_low=ci.low,ci_high=ci.high))
write(pd.DataFrame(rows),'pre_post_correlations')
# Training: criterion/cap differences are descriptive, not randomised dose comparisons.
tr=pd.read_csv(ROOT/'inputs/training_summary.csv').set_index('participant')
tr['fb_pre']=fw['before','optimised'];tr['fb_post']=fw['after','optimised'];tr['fb_change']=tr.fb_post-tr.fb_pre
tr['speech_pre']=sep['before','optimised'];tr['speech_post']=sep['after','optimised'];tr['speech_change']=tr.speech_post-tr.speech_pre
tr['srm_pre']=w['before','optimised'];tr['srm_post']=w['after','optimised'];tr['srm_change']=tr.srm_post-tr.srm_pre
write(tr.reset_index(),'training_outcomes')
write(tr.groupby('stop')[['n_trials','duration_min','successful_blocks','fb_pre','fb_post','fb_change','speech_pre','speech_post','speech_change','srm_pre','srm_post','srm_change']].mean().reset_index(),'training_group_summary')
rows=[]
for x in ['n_trials','digit_accuracy_pct']:
 for y in ['fb_change','speech_change','srm_change']:
  rr=stats.spearmanr(tr[x],tr[y]);rows.append(dict(training_measure=x,outcome=y,rho=rr.statistic,p=rr.pvalue,n=18))
r=pd.DataFrame(rows);r['p_holm']=multipletests(r.p,method='holm')[1];write(r,'training_associations')
# Trial count integrity and deterministic selection.
write(srm.groupby(['participant','session','hrtf','configuration']).size().rename('n').reset_index(),'speech_cell_counts')
r=pd.read_csv(ROOT/'inputs/srm_trial_identity.csv');r['correct']=r.correct.astype(int)
assert len(r)==len(srm)
assert np.array_equal(r.groupby(['participant','session','hrtf','configuration']).correct.sum().sort_index(),srm.groupby(['participant','session','hrtf','configuration']).correct.sum().sort_index())
# Timing audit: low-frequency (200-1500 Hz) band-limited cross-correlation, 16x grid.
timing=np.load(ROOT/'inputs/optimisation_timing_inputs.npz');fs=float(timing['fs']);base=timing['baseline_spectrum'];optir=timing['optimised_ir']
baseir=np.fft.irfft(base,n=256,axis=-1)
def itd(ir):
 n=8192;freq=np.fft.rfftfreq(n,1/fs);H=np.fft.rfft(ir,n,axis=-1)
 cross=H[0]*H[1].conj();cross[(freq<200)|(freq>1500)]=0
 cc=np.fft.irfft(cross,n*16);lag=np.arange(-int(.001*fs*16),int(.001*fs*16)+1)
 return float(lag[np.argmax(cc[lag%(n*16)])]/(fs*16)*1e6)
rows=[]
for j,az in enumerate([30,150]):
 opt=np.fft.rfft(optir[j],256);change=np.angle((opt[0]*opt[1].conj())/(base[j,0]*base[j,1].conj()));f=np.fft.rfftfreq(256,1/fs)
 rows.append(dict(azimuth=az,baseline_itd_us=itd(baseir[j]),optimised_itd_us=itd(optir[j]),delta_itd_us=itd(optir[j])-itd(baseir[j]),max_ipd_change_200_1500_rad=np.max(abs(change[(f>=200)&(f<=1500)]))))
write(pd.DataFrame(rows),'timing_audit')

# Phase and cross-correlation checks for the AES export plotted in Figure 1.
with np.load(ROOT / 'inputs/aes_log_optimisation.npz') as aes, np.load(ROOT / 'inputs/acoustic_inputs.npz') as plotted:
    phase_rows = []
    for j, azimuth in enumerate((30, 150)):
        index = np.flatnonzero(aes['az_deg'] == azimuth).item()
        original = np.stack([aes['H_L_original'][index], aes['H_R_original'][index]])
        optimised = plotted['aes_log_ir'][j].astype(float)
        reference_itd = itd(np.fft.irfft(original, n=optimised.shape[-1]))
        optimised_itd = itd(optimised)
        frequencies = np.fft.rfftfreq(optimised.shape[-1], 1 / fs)
        within_band = (frequencies >= 200) & (frequencies <= 8000)
        phase_error = np.angle(np.fft.rfft(optimised) * original.conj())
        phase_rows.append(dict(
            azimuth=azimuth,
            baseline_peak_lag_us=reference_itd,
            aes_peak_lag_us=optimised_itd,
            delta_peak_lag_us=optimised_itd - reference_itd,
            max_abs_phase_error_200_8000_rad=np.max(np.abs(phase_error[:, within_band])),
        ))
write(pd.DataFrame(phase_rows), 'aes_timing_audit')

for name in ['h1','front_back_summary','front_back_binomial_sensitivity','front_back_permutation','speech_changes','behavioural_srm','within_configuration_correlations','configuration_adjusted_gee','pre_post_correlations','training_group_summary','training_associations','timing_audit']:
 print('\n'+name+'\n'+pd.read_csv(OUT/(name+'.csv')).to_string(index=False))
