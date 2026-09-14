"""Match global entropy using all stored logits, with no teacher queries."""
import argparse
import json
from pathlib import Path
import torch


def load(root):
    chunks = []
    for e in range(400):
        for b in range(15):
            z = torch.load(root / f'epoch_{e}/batch_{b}.tar', map_location='cpu', weights_only=False)[5].float()
            assert z.shape == (20, 100) and torch.isfinite(z).all()
            chunks.append(z)
    return torch.cat(chunks)


def entropy(z, t):
    logp = (z / t).log_softmax(1)
    return float((-(logp.exp() * logp).sum(1)).double().mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    target = entropy(load(a.reference), 20.)
    z = load(a.candidate)
    lo, hi = .01, 1000.
    assert entropy(z, lo) < target < entropy(z, hi)
    for _ in range(45):
        mid = (lo + hi) / 2
        if entropy(z, mid) < target:
            lo = mid
        else:
            hi = mid
    t = (lo + hi) / 2
    matched = entropy(z, t)
    assert abs(matched - target) < 1e-6
    result = dict(status='complete', views=120000, reference=str(a.reference), candidate=str(a.candidate),
                  reference_temperature=20., student_temperature=20., teacher_temperature=t,
                  reference_entropy=target, candidate_entropy_T20=entropy(z, 20.), matched_entropy=matched,
                  entropy_units='natural_log', new_teacher_queries=0)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
