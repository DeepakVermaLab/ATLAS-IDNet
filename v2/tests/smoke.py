import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from atlas_v2.data import prepare
from atlas_v2.experiments import run, diagnostics, summarize
from atlas_v2.agent import EvidenceAgent

with tempfile.TemporaryDirectory() as temp:
    root=Path(temp); rng=np.random.default_rng(11)
    frame=pd.DataFrame(rng.normal(size=(180,6)),columns=['bytes','packets','tcp_flag','protocol','duration','iat'])
    frame['label']=np.tile(['benign','attack','other'],60)
    frame.to_csv(root/'raw.csv',index=False)
    spec={'csv_globs':[str(root/'raw.csv')],'label':'label','exclude_columns':[], 'max_rows':180,'split':'stratified'}
    prepare(spec,root/'prepared')
    d=np.load(root/'prepared/data.npz')
    assert not set(d['ids_train']) & set(d['ids_test'])
    for variant in ['atlas','legacy','no_global_context','mlp','single_residual','no_residual','mean','concat','random_groups','random_forest','extra_trees','hist_gradient','logistic']:
        out=run(root/'prepared',root/'runs',variant=variant,epochs=1,batch_size=32,width=8)
        assert (out/'complete.json').exists()
    out=run(root/'prepared',root/'runs',epochs=1,batch_size=32,width=8)
    timestamp=(out/'complete.json').stat().st_mtime_ns
    assert run(root/'prepared',root/'runs',epochs=1,batch_size=32,width=8)==out
    assert timestamp==(out/'complete.json').stat().st_mtime_ns
    (out/'complete.json').unlink()
    run(root/'prepared',root/'runs',epochs=1,batch_size=32,width=8)
    assert len(pd.read_csv(out/'history.csv'))==1
    for loss in ['balanced','focal']:
        run(root/'prepared',root/'runs',epochs=1,batch_size=32,width=8,loss=loss)
    diagnostics(root/'prepared',out,samples=8,repeats=3)
    resources=json.loads((out/'resource_profile.json').read_text())
    assert resources['device']=='cpu'
    assert resources['model_parameter_bytes']>0
    assert resources['model_file_bytes']>0
    assert resources['trainable_parameters']>0
    assert resources['linear_macs_per_flow']>0
    report=EvidenceAgent(out).run(out/'agent')
    assert report['verification']['unsupported_claim_rate']==0
    assert len(summarize(root/'runs'))>0
    print('PASS: all variants, losses, completion reuse, checkpoint resume, diagnostics, agent, summary')
