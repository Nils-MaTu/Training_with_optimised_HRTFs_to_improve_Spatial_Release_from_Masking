import pandas as pd,numpy as np
from pathlib import Path
from scipy import stats
import statsmodels.api as sm,statsmodels.formula.api as smf
O=Path(__file__).resolve().parent
d=pd.read_csv(O/'inputs/srm_trial_identity.csv');d.correct=d.correct.astype(int)
rows=[]
for cfg,g in d[d.session=='before'].groupby('configuration'):
 g=g.copy();g['opt']=(g.hrtf=='optimised').astype(int);g['order']=g.total_trial-g.groupby('participant').total_trial.transform('mean')
 f=smf.gee('correct ~ opt + order + C(trial_id)',groups='participant',data=g,family=sm.families.Binomial(),cov_struct=sm.cov_struct.Independence()).fit(cov_type='bias_reduced')
 b=f.params['opt'];se=f.bse['opt'];q=stats.t.ppf(.975,17)
 paired=g.pivot_table(index=['participant','trial_id'],columns='hrtf',values='total_trial').dropna()
 rows.append(dict(configuration=cfg,n=len(g),pct_individual_first=100*(paired.individual<paired.optimised).mean(),OR=np.exp(b),OR_ci_low=np.exp(b-q*se),OR_ci_high=np.exp(b+q*se),p=2*stats.t.sf(abs(b/se),17)))
r=pd.DataFrame(rows);r.to_csv(O/'results/order_sensitivity.csv',index=False);print(r.to_string(index=False))
