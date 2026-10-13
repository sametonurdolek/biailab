"""Consolidate confirmed files; no model inference or new scoring."""
import json,sys
from pathlib import Path
from common import ROOT,load_json,save_json
from audit_dataset import digest

def main():
    audit=load_json('pairs.json');split=load_json('split.json');cal=load_json('calibration.json');env=load_json('environment.json')
    metrics=json.loads((ROOT/'results'/'metrics.json').read_text());reps=json.loads((ROOT/'results'/'representatives.json').read_text())
    ckpts={name:json.loads((ROOT/'checkpoints'/name/'completed.json').read_text()) for name in ['scalar','spatial']}
    means=metrics['means']
    text='## Confirmed results\n\n'
    text+='Completed paired-document experiment, one seed. These numbers come from `results/metrics.json` -> `means`; per-image rows are in `results/per_image.csv`.\n\n'
    text+=f"Valid pairs: {len(audit['pairs'])}; unmatched: {len(audit['unmatched'])}; content groups: {len(split['content_groups'])}; geometric registrations: {sum(r['registration'] for r in split['prepared'])}.\n\n"
    text+='Split: '+', '.join(f'{s}={len(v)}' for s,v in split['partitions'].items())+'. Exact identities: `manifests/split.json` -> `partitions`.\n\n'
    text+=f"Shared pipeline: `{cal['pipeline']['name']}`. Bandwidth interval [{cal['h_lo']:.10g}, {cal['h_hi']:.10g}], fixed-h calibration optimum h*={cal['h_star']:.10g}; endpoint optimum={cal['endpoint_optimum']}. NLM patch7/search21. See `manifests/calibration.json`.\n\n"
    text+='| System | Mean PSNR (dB) | Mean SSIM |\n|---|---:|---:|\n'
    text+=f"| Scalar | {means['scalar_psnr']:.6f} | {means['scalar_ssim']:.6f} |\n| Spatial | {means['spatial_psnr']:.6f} | {means['spatial_ssim']:.6f} |\n| Paired spatial − scalar | {means['delta_psnr']:+.6f} | {means['delta_ssim']:+.6f} |\n\n"
    text+='Spatial has higher mean PSNR and lower mean SSIM in this completed run. The measurements do not show dominance across both metrics. No parameters were changed after this test observation.\n\n'
    for name in ['scalar','spatial']:
        c=ckpts[name];text+=f"{name.capitalize()} checkpoint: `checkpoints/{name}/best.pt`, best validation step {c['best_step']}, validation PSNR {c['best_val_psnr']:.6f}. Both completed {c['steps']} updates. SHA256: `{c['checkpoint_sha256']}`.\n\n"
    text+='| Image | Scalar PSNR | Spatial PSNR | ΔPSNR | Scalar SSIM | Spatial SSIM | ΔSSIM |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for r in metrics['per_image']:
        text+=f"| {r['image']} | {r['scalar_psnr']:.6f} | {r['spatial_psnr']:.6f} | {r['delta_psnr']:+.6f} | {r['scalar_ssim']:.6f} | {r['spatial_ssim']:.6f} | {r['delta_ssim']:+.6f} |\n"
    text+='\nRepresentative figures (first / best / upper-median / worst rule, deduplicated):\n\n'
    for r in reps['artifacts']:text+=f"* [{r['image']} comparison]({Path(r['comparison']).as_posix()}) and [closeups]({Path(r['closeups']).as_posix()}).\n"
    text+=f"* [Precommitted diagram crop]({reps['diagram_example']['path']}).\n"
    text+='\nEvery test image also has a comparison figure, float outputs, h maps and absolute errors.\n\n'
    failures=[]
    for p in (ROOT/'logs').glob('*.status.json'):
        m=json.loads(p.read_text())
        if m['exit_code']:failures.append(m)
    text+='Recorded phase failures: '+str(len(failures))+'. Dataset exclusions: '+str(len(split['excluded']))+'.\n\n'
    text+='A preliminary console filename listing hit a Turkish-character encoding error; it was rerun with UTF-8 before the dataset audit. No training/evaluation failure or OOM was recorded.\n\n'
    text+=f"Environment: Python {env['python'].split()[0]}, Torch {env['pytorch']}, CUDA {env['cuda']}, OpenCV {env['opencv']}, {env['gpu']}. See `manifests/environment.json`.\n\n"
    text+='The mean paired difference is the measured contrast on this prepared dataset and seed. It does not establish general document superiority, seed stability, color fidelity or production behavior. Target-guided geometry, CamScanner-generated targets and repeated/source-dependent question content remain limitations.\n'
    path=ROOT/'README.md';old=path.read_text(encoding='utf-8');old=old.split('## Confirmed results')[0]+text
    old=old.replace('Status: dataset audit and shared implementation prepared; calibration/training/test\nresults will be recorded below only after their phases complete.','Status: calibration, both 400-step training budgets, frozen test evaluation and\nvisualization completed. Verified results and artifact paths are below.')
    path.write_text(old,encoding='utf-8')
    save_json('results/final_report.json',dict(valid_pairs=len(audit['pairs']),content_groups=len(split['content_groups']),
        split={s:len(v) for s,v in split['partitions'].items()},registration_pairs=sum(r['registration'] for r in split['prepared']),
        calibration=dict(pipeline=cal['pipeline'],h_lo=cal['h_lo'],h_hi=cal['h_hi'],h_star=cal['h_star'],endpoint_optimum=cal['endpoint_optimum']),
        checkpoints=ckpts,means=means,representative_figures=reps['artifacts'],diagram_figure=reps['diagram_example'],failures=failures,exclusions=split['excluded']))
    print(json.dumps(means,indent=2))

if __name__=='__main__':main()
