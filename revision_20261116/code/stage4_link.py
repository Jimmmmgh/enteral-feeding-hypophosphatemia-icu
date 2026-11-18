"""Read-only replay of frozen source cohort for internal revision linkage."""
import ast
import hashlib
import json
from pathlib import Path
from types import ModuleType
import sys

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
REVISION = Path(__file__).resolve().parents[1]
PROJECT = REVISION.parent
OUT = REVISION / 'results' / 'stage4'
SIC = PROJECT.parents[1] / 'SICdb' / '1.0.8'


def extractor():
    """Use original implementation through cohort assembly and exit before all file writes."""
    path = PROJECT / 'scripts' / 'extract.py'
    document = ast.parse(path.read_text(encoding='utf-8'))
    process = next(node for node in document.body if isinstance(node, ast.FunctionDef) and node.name == 'process')
    paths = next(node for node in document.body if isinstance(node, ast.Assign) and ast.unparse(node).startswith("A = ROOT / 'audit'"))
    document.body.remove(paths)
    process.body = [
        node for node in process.body
        if not (isinstance(node, ast.Expr) and ast.unparse(node).startswith('counts.to_csv('))
    ]
    end = next(i for i, node in enumerate(process.body) if ast.unparse(node).startswith('b = b.drop(columns='))
    process.body[end:] = [ast.Return(ast.Tuple([ast.Name('b', ast.Load()), ast.Name('qa', ast.Load())], ast.Load()))]
    module = ModuleType('frozen_cohort_read_only')
    module.__file__ = str(path)
    sys.path.insert(0, str(path.parent))
    exec(compile(ast.fix_missing_locations(document), str(path), 'exec'), module.__dict__)
    module.A = PROJECT / 'audit'
    return module


def replay(module, source):
    inputs = getattr(module, 'source_' + source)()
    cohort, replay_audit = module.process(source, *inputs)
    audit = json.loads((PROJECT / 'audit' / f'{source}_extraction.json').read_text(encoding='utf-8'))
    feature_file = PROJECT / 'private' / f'{source}_features.parquet'
    if hashlib.sha256(feature_file.read_bytes()).hexdigest() != audit['feature_file_sha256']:
        raise RuntimeError(f'{source} frozen feature file digest differs')
    if replay_audit['flow'] != audit['flow']:
        raise RuntimeError(f'{source} reproduced cohort flow differs, stop linkage')
    cols = pd.read_parquet(feature_file)
    check = cohort.reset_index(drop=True).loc[:, cols.columns]
    pd.testing.assert_frame_equal(check, cols, check_dtype=False, check_exact=False, rtol=1e-9, atol=1e-10)
    if not cohort.sid.is_unique:
        raise RuntimeError(f'{source} reconstructed source identity is not unique')
    cohort = cohort.reset_index(drop=True)
    return cohort, inputs[0]


def grouped_phosphate(events, cohort, relative_scale):
    subset = events.merge(cohort[['sid', 'index_hours', 'end_hours']], on='sid', how='inner', validate='many_to_one')
    start = subset['time'] / relative_scale
    base = start.gt(subset.index_hours - 24) & start.le(subset.index_hours)
    post = start.gt(subset.index_hours) & start.le(np.minimum(subset.end_hours, subset.index_hours + 48))
    result = {}
    for label, mask in [('pre', base), ('post', post)]:
        for route in ('iv', 'enteral', 'unknown', 'any'):
            valid = mask & (subset.route.eq(route) if route != 'any' else True)
            result[f'phos_{label}_{route}'] = cohort.sid.isin(subset.loc[valid, 'sid']).to_numpy()
    return result
