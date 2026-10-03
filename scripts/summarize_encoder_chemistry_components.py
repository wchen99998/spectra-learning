"""Summarize validation-selected component confirmation without selecting new tests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path('artifacts/encoder_chemistry_20260930')
ARMS=('parent_1190k','periodic_1209k')


def main():
    output=ROOT/'component_analysis'
    output.mkdir(exist_ok=True)
    summary={'scope':'Three branch layers per kind selected on256 validation molecules, then tested on512 separate test molecules. All8heads in selected attention layers tested. Largest observed effects below are descriptive rankings within this fixed tested set; no claim of independent discovery for test-best component.',
             'intervals':'Unadjusted paired molecule bootstrap95% intervals, conditional on fixed decoders; correlated interventions and unequal perturbation sizes.', 'arms':{}}
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained',width_ratios=[1.4,1.3,1])
    summaries=[json.loads((ROOT/a/'causal/components_test_512_selected/analysis.json').read_text()) for a in ARMS]
    head_limit=max(abs(r['groups']['all']['delta_balanced_bce']) for s in summaries for r in s['results'] if r['variant'].startswith('head_'))
    for row,(arm,s) in enumerate(zip(ARMS,summaries)):
        assert s['folds']==['test'] and s['molecules']==512
        path=ROOT/arm/'causal/components_test_512_selected'
        m=json.loads((path/'manifest.json').read_text())
        assert m['predictions_sha256']==hashlib.sha256((path/'predictions.npz').read_bytes()).hexdigest()
        assert all(p['bitwise_equal'] for p in m['baseline_cache_parity'])
        assert not m['selection']['test_used']
        entries=[]
        largest={}
        for r in s['results'][1:]:
            v=r['groups']['all']
            entries.append({'variant':r['variant'],'delta':v['delta_balanced_bce'],'ci95':v['delta_bce_ci95'],'delta_auroc':v['delta_auroc']})
        for kind in ('attention','mlp','metadata','head'):
            largest[kind]=max((e for e in entries if e['variant'].startswith(kind+'_')),key=lambda e:e['delta'])
        summary['arms'][arm]={'molecules':512,'supported_targets':s['results'][0]['groups']['all']['targets'],
             'validation_selection':m['selection'],'largest_observed':largest,'conditions':entries,
             'baseline_balanced_bce':s['results'][0]['groups']['all']['balanced_bce'],
             'baseline_auroc':s['results'][0]['groups']['all']['auroc'],'parity_all_bitwise':True}
        branch=[e for e in entries if e['variant'].startswith(('attention_','mlp_','metadata_'))]
        y=np.arange(len(branch))
        values=np.array([e['delta'] for e in branch]); ci=np.array([e['ci95'] for e in branch])
        axes[row,0].errorbar(values,y,xerr=np.maximum(0,np.stack([values-ci[:,0],ci[:,1]-values])),fmt='o',color='#297a8b',capsize=2)
        axes[row,0].set_yticks(y,[e['variant'].replace('_scale0','').replace('_block',' · block ') for e in branch],fontsize=8)
        axes[row,0].invert_yaxis(); axes[row,0].axvline(0,color='#888888',lw=.7)
        axes[row,0].set(title=arm.replace('_',' '),xlabel='Change in balanced BCE')
        layers=m['head_layers']; matrix=np.zeros((len(layers),8))
        for e in entries:
            if e['variant'].startswith('head_block'):
                layer,head=e['variant'].removeprefix('head_block').removesuffix('_scale0').split('_head')
                matrix[layers.index(int(layer)),int(head)]=e['delta']
        im=axes[row,1].imshow(matrix,cmap='RdBu_r',vmin=-head_limit,vmax=head_limit,aspect='auto')
        axes[row,1].set(xlabel='Ablated head (0-based)',ylabel='Validation-selected block',title='Individual head effects')
        axes[row,1].set_xticks(range(8));axes[row,1].set_yticks(range(3),layers)
        for i in range(3):
            for j in range(8):
                axes[row,1].text(j,i,f'{matrix[i,j]:.3f}',ha='center',va='center',fontsize=7,color='white' if abs(matrix[i,j])>.7*head_limit else '#182c34')
        dose=[e for e in entries if e['variant'].startswith('final_metadata_')]
        x=np.array([float(e['variant'].removeprefix('final_metadata_scale')) for e in dose]);v=np.array([e['delta'] for e in dose]);c=np.array([e['ci95'] for e in dose])
        axes[row,2].errorbar(np.r_[x,1],np.r_[v,0],yerr=np.c_[np.maximum(0,np.stack([v-c[:,0],c[:,1]-v])),[0,0]],marker='o',color='#9f662d',capsize=2)
        axes[row,2].set(xlabel='Final modulation retained',ylabel='Change in balanced BCE',title='Final affine dose response')
        axes[row,2].axhline(0,color='#888888',lw=.7)
        for ax in (axes[row,0],axes[row,2]):ax.spines[['top','right']].set_visible(False)
    fig.colorbar(im,ax=axes[:,1],label='Change in balanced BCE',shrink=.6)
    fig.suptitle('Frozen-decoder component necessity · independent512-molecule test cohort')
    for suffix in ('png','pdf'):fig.savefig(output/f'component_heads.{suffix}',dpi=180)
    plt.close(fig)
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    for a,v in summary['arms'].items():print(a,v['largest_observed'])


if __name__=='__main__':main()
