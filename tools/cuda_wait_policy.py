"""Validate the CUDA wait helper startup receipt before accepting a run."""
import re


def verify_wait_policy(stderr, requested):
    rows=re.findall(r'^LLAMA_CUDA_WAIT device=(\d+) policy=(auto|blocking) flags=(\d+) verified=1$',stderr,re.M)
    expected=4 if requested=='blocking' else 0
    if requested not in ('auto','blocking') or len(rows)!=1:
        raise ValueError('Missing or ambiguous CUDA wait-policy receipt')
    device,policy,flags=rows[0]
    if device!='0' or policy!=requested or int(flags)&7!=expected:
        raise ValueError('CUDA wait-policy readback mismatch')
    return {'device':int(device),'policy':policy,'flags':int(flags)}
