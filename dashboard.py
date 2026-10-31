"""DualNet-AL results dashboard:  streamlit run dashboard.py"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config('DualNet-AL', layout='wide')
st.title('DualNet-AL: active-learning results')

runs = sorted((Path(__file__).parent / 'results').glob('*.json'), reverse=True)
if not runs:
    st.info('No results yet. Run `python dualnet_al.py` first.')
    st.stop()
run = st.sidebar.selectbox('Run', runs, format_func=lambda p: p.stem)
res = json.loads(run.read_text())
st.sidebar.json(res['config'], expanded=False)

rows = [dict(strategy=s, seed=k, **{m: h[m] for m in ('round', 'n', 'acc', 'macro_f1', 'tail_f1', 'imbalance')})
        for s, seeds in res['results'].items() for k, hist in enumerate(seeds) for h in hist]
mean = pd.DataFrame(rows).groupby(['strategy', 'n']).mean(numeric_only=True).reset_index()

for col, (m, label) in zip(st.columns(3), [('acc', 'Test accuracy vs labels'),
                                           ('tail_f1', 'Tail-class F1 vs labels'),
                                           ('imbalance', 'Labelled-pool imbalance (max/min)')]):
    col.subheader(label)
    col.line_chart(mean.pivot(index='n', columns='strategy', values=m))

st.subheader('Summary (mean ± std over seeds)')
summary = pd.DataFrame({s: {m: f'{mu:.3f} ± {sd:.3f}' for m, (mu, sd) in d.items()}
                        for s, d in res['summary'].items()}).T
st.dataframe(summary, use_container_width=True)

st.subheader('Per-class view')
r = st.slider('Round', 0, res['config']['rounds'], res['config']['rounds'])
per_class = lambda key: pd.DataFrame({s: np.mean([h[r][key] for h in seeds], 0)
                                      for s, seeds in res['results'].items()})
c1, c2 = st.columns(2)
c1.caption('Labels per class in the labelled pool (class 0 = most frequent in the pool)')
c1.bar_chart(per_class('counts'), stack=False)
c2.caption('Per-class F1 on the balanced test set')
c2.bar_chart(per_class('f1'), stack=False)

st.download_button('Download summary CSV', summary.to_csv().encode(), f'{run.stem}_summary.csv')
st.download_button('Download raw results JSON', run.read_bytes(), run.name)
