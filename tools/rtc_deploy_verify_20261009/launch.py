from pathlib import Path
import os,subprocess,json
root=Path('/gaozt-test1/fyhong/OpenWAM');out=root/'outputs/rtc_deploy_verify_20261009';py='/gaozt-test1/fyhong/envs/statewam/bin/python'
gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])
for model in ['new','old']:
 free=[int(x) for x in subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits']).decode().splitlines()]
 if free[gpu]<29500:raise RuntimeError('Insufficient free VRAM for next diagnostic model')
 with (out/(model+'.log')).open('x') as f:
  result=subprocess.run([py,'-u',str(root/'tools/rtc_deploy_verify_20261009/run.py'),'--model',model,'--out',str(out)],cwd=root,stdout=f,stderr=subprocess.STDOUT)
 if result.returncode:raise RuntimeError('Diagnostic failed: '+model)
(out/'all_complete.json').write_text(json.dumps({'models':['new','old']}))
