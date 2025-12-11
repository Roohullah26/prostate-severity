import torch
from pathlib import Path
import sys

p = Path('models/baseline_real_t2_adc_3s_ep1.pth')
if not p.exists():
    print('File not found:', p)
    sys.exit(1)

ckpt = torch.load(p, map_location='cpu')
print('type:', type(ckpt))
if isinstance(ckpt, dict):
    print('top-level keys:', list(ckpt.keys())[:20])
    # try to find a state_dict
    for k in ['state_dict','model_state','model_state_dict','model']:
        if k in ckpt:
            sd = ckpt[k]
            print('\nFound',k,'with',len(sd),'keys. Sample keys:')
            for i,k2 in enumerate(list(sd.keys())[:50]):
                print(i, k2)
            break
    else:
        # assume ckpt itself is a state dict
        print('\nTreating ckpt as state_dict with', len(ckpt.keys()), 'keys. Sample:')
        for i,k2 in enumerate(list(ckpt.keys())[:50]):
            print(i, k2)
else:
    print('Checkpoint is not a dict; cannot inspect')
