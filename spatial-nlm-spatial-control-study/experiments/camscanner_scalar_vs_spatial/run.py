"""Run one phase with exclusive durable logs; errors never disappear into a rerun."""
import argparse,os,subprocess,sys,json,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['audit_dataset','prepare_pairs','verify','calibrate','train','evaluate','visualize','report']);args=parser.parse_args()
    logs=ROOT/'logs';logs.mkdir(exist_ok=True)
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    path=logs/f'{stamp}_{args.stage}.log';env=os.environ.copy();env['PYTHONIOENCODING']='utf-8';env['PYTHONHASHSEED']='20261014'
    with path.open('x',encoding='utf-8') as log:
        proc=subprocess.Popen([sys.executable,'-u',str(ROOT/(args.stage+'.py'))],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
        for line in proc.stdout:log.write(line);log.flush();print(line,end='',flush=True)
        rc=proc.wait();log.write(f'\nEXIT_CODE={rc}\n')
    path.with_suffix('.status.json').write_text(json.dumps(dict(stage=args.stage,exit_code=rc,log=str(path),timestamp_utc=stamp),indent=2))
    raise SystemExit(rc)

if __name__=='__main__':main()

