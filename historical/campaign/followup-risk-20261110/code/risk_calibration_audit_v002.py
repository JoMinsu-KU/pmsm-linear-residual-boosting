"""Independent reconstruction of C16 symmetric per-loss shift calibration.

No new producer import, training, pickle execution or fresh-confirmation claim.
The immutable v001 independent audit supplies generic lineage and risk metrics.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
sys.dont_write_bytecode = True
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FOLLOWUP = ROOT / 'followup-risk-20261110'
SHARED_SHA = '2cfb48953741515e8d9815073075bcc3e46c717c5110837d280acdb93a2bd035'
PROTOCOL_SHA = '202554d6723531e76ab8210a7e7f63048fb0ba14ec38187685ae1bdb969f9260'
METRICS_PRODUCER_SHA = '7d514ccd7f9471a4e5d8dcc29604d8782c6c0dad300cdc830a253d4e1c759282'
MLP_PRODUCER_SHA = '54ec228ba97251b1b427845a57c8bd0f906729cab4d950e798eb8ffd9bc5cd7f'
SEEDS = [2026111701, 2026111702, 2026111703]
LOSSES = ['mse', 'asym4', 'q90']
SHIFT_GRID = [0., .1, .2, .3, .4, .5, .75, 1., 1.25, 1.5, 2., 2.5, 3., 4., 5., 7.5, 10.]
QUANTILES = [.5, .6, .7, .8, .9, .95]
RMSE_RATIOS = [1., 1.02, 1.05, 1.10]
OVER_RATIOS = [1., 1.10, 1.25, 1.50]
CONTEXT_KEYS = {'prediction__direct9_fitcv_selected', 'prediction__positive_rc_insample_boost',
                'prediction__linear33_insample_boost'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def local(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Path outside canonical campaign')
    return path


def dependency():
    path = FOLLOWUP / 'code/risk_audit_v001.py'
    if sha(path) != SHARED_SHA:
        raise ValueError('Frozen independent v001 audit changed')
    spec = importlib.util.spec_from_file_location('c16_symmetric_independent_dependency', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit_binds(a, report, entry, label):
    path = local(entry['path'])
    suffix = path.relative_to(ROOT).as_posix()
    hits = [v for p, v in report['inputs'].items() if p.replace('\\', '/').endswith('/' + suffix)]
    a.require(bool(hits) and all(h == entry['sha256'] for h in hits), label)


def load_components(a, helper, cfg, inputs, config_sha):
    source = helper.existing_role_records(a, cfg)
    roles = [r['role'] for r in cfg['role_inputs']]
    role_ids = {r['role']: r['ids'] for r in cfg['role_inputs']}
    coverage = [(r, pid) for r in roles for pid in role_ids[r]]
    components, metadata, reports = {}, {}, {}
    for component in inputs['components']:
        cid = component['id']
        a.require(cid not in components, 'Unique source component')
        a.require(component['backbone'] in ('mlp', 'hgb', 'context'), 'Declared source backbone')
        a.require([(p['role'], p['profile']) for p in component['predictions']] == coverage,
                  'Exact source role/profile coverage/' + cid)
        if component['backbone'] == 'mlp':
            report = read(a.pin(component['audit']))
            a.require(report.get('schema') == 'c16-risk-independent-audit-v001'
                      and report.get('status') == 'PASS' and report.get('mode') == 'mlp'
                      and report.get('auditor_sha256') == SHARED_SHA
                      and report.get('producer_sha256') == MLP_PRODUCER_SHA
                      and report.get('config_sha256') == config_sha and report.get('failures') == [],
                      'Original complete independent MLP forward audit/' + cid)
            a.pin(dict(path='followup-risk-20261110/code/risk_mlp_v001.py', sha256=MLP_PRODUCER_SHA))
            result = report['results']
            a.require(result['dataset'] == cfg['dataset'] and result['loss'] == component['loss']
                      and result['seed'] == component['seed'] and result['epochs'] == 200,
                      'Forward audit dataset/loss/seed/epochs identity/' + cid)
            a.require(result['all_evaluation_previously_exposed'] is True
                      and result['independent_confirmation'] is False,
                      'Original forward audit exploratory scope/' + cid)
            receipt_path = a.pin(component['source_receipt'])
            a.require(not (receipt_path.parent / 'failure.json').exists(), 'No failed original neural source')
            receipt = read(receipt_path)
            a.require(receipt['status'] == 'MLP_PREDICTIONS_LOCKED' and receipt['completed_comparator'] is True
                      and receipt['source_sha256'] == MLP_PRODUCER_SHA and receipt['config_sha256'] == config_sha
                      and receipt['dataset'] == cfg['dataset'] and receipt['loss'] == component['loss']
                      and receipt['seed'] == component['seed'], 'Exact original neural receipt identity')
            audit_binds(a, report, component['source_receipt'], 'Original forward audit binds neural receipt')
            a.same(receipt['predictions'], component['predictions'], 'Neural receipt exact prediction panel')
            lock_path = a.pin(receipt['prediction_lock'])
            audit_binds(a, report, receipt['prediction_lock'], 'Forward audit binds original prediction lock')
            lock = read(lock_path)
            a.require(lock['truth_accessed'] is False and lock['config_sha256'] == config_sha
                      and lock['source_sha256'] == MLP_PRODUCER_SHA, 'Original neural prediction lock identity')
            a.same(lock['predictions'], component['predictions'], 'Neural lock exact prediction panel')
            for entry in component['predictions']:
                audit_binds(a, report, entry, 'Forward audit binds exact prediction/' + cid)
            reports[cid] = report
        elif component['backbone'] == 'hgb':
            receipt = a.pin(component['source_receipt'])
            a.require(not (receipt.parent / 'failure.json').exists(), 'No failed tree source')
            tree = read(receipt)
            a.require(tree['status'] == 'HGB_FIT_PREDICTIONS_LOCKED'
                      and tree['dataset'] == cfg['dataset'] and tree['config_sha256'] == config_sha
                      and tree['loss'] == component['loss'] and tree['seed'] == component['seed'],
                      'Exact tree source receipt')
            a.pin(dict(path='followup-risk-20261110/code/risk_tree_v001.py', sha256=tree['source_sha256']))
            for name, digest in tree['files'].items():
                a.pin(dict(path=name, sha256=digest), receipt.parent)
            a.require('prediction_lock.json' in tree['files'], 'Tree lock is receipt-bound')
            lock = read(receipt.parent / 'prediction_lock.json')
            a.require(lock['config_sha256'] == config_sha and lock['truth_accessed'] is False,
                      'Tree same-config target-free prediction lock')
            a.same(tree['predictions'], component['predictions'], 'Exact tree receipt predictions')
            a.same(lock['predictions'], component['predictions'], 'Exact tree locked predictions')
        values = {r: {} for r in roles}
        for entry in component['predictions']:
            record = source[(entry['role'], entry['profile'])]
            saved = a.arrays(a.pin(entry))
            index_key = entry.get('index_key', 'index')
            prediction_key = entry.get('prediction_key', 'prediction')
            a.require(np.array_equal(saved[index_key], record['index']), 'Prediction/target endpoint identity/' + cid)
            prediction = saved[prediction_key].astype(np.float64)
            a.require(prediction.shape == record['truth'].shape, 'Complete scalar prediction/' + cid)
            if 'source_row' in saved:
                row = saved['source_row']
                if len(row) != len(record['source_row']):
                    row = row[record['index']]
                a.require(np.array_equal(row, record['source_row']), 'Prediction raw-row lineage/' + cid)
            if component['backbone'] == 'context':
                a.require(prediction_key in CONTEXT_KEYS, 'Predeclared contextual identity')
                a.require(np.array_equal(prediction, record['source'][prediction_key]), 'Unchanged contextual prediction')
            values[entry['role']][entry['profile']] = prediction
        components[cid], metadata[cid] = values, component
    seeds = sorted({c['seed'] for c in metadata.values() if c['backbone'] == 'mlp'})
    a.require(seeds in ([SEEDS[0]], SEEDS), 'Complete first-seed check or all three registered neural seeds')
    for seed in seeds:
        group = [c for c in metadata.values() if c['backbone'] == 'mlp' and c['seed'] == seed]
        a.require(len(group) == 3 and {c['loss'] for c in group} == set(LOSSES), 'All paired MLP losses')
        for field in ('initial_array_sha256', 'training_array_sha256', 'permutation_digests'):
            items = [reports[c['id']]['results'][field] for c in group]
            a.require(all(item == items[0] for item in items), 'Preserved paired initialization/data/order/' + field)
    trees = [c for c in metadata.values() if c['backbone'] == 'hgb']
    a.require(len(trees) == 2 and {c['loss'] for c in trees} == {'mse', 'q90'}
              and all(c['seed'] == SEEDS[0] for c in trees), 'Exactly two fixed tree controls')
    contexts = [c for c in metadata.values() if c['backbone'] == 'context']
    a.require(len(contexts) == 3 and all(c['loss'] == 'context' and c['seed'] is None for c in contexts),
              'Three unselected contextual controls')
    a.require({c['predictions'][0].get('prediction_key') for c in contexts} == CONTEXT_KEYS,
              'Exact contextual control panel')
    for c in contexts:
        a.require(len({e.get('prediction_key') for e in c['predictions']}) == 1, 'One context identity across roles')
    return source, components, metadata, seeds


def shifts_for(profiles):
    """Profile-balanced empirical inverse CDF; stable sort, first duplicate kept."""
    errors = np.concatenate([y - p for _, y, p, _ in profiles])
    weights = np.concatenate([np.full(len(y), 1. / (len(profiles) * len(y))) for _, y, _, _ in profiles])
    order = np.argsort(errors, kind='stable')
    sorted_errors = errors[order]
    cdf = np.cumsum(weights[order])
    shifts = list(SHIFT_GRID)
    for q in QUANTILES:
        index = min(int(np.searchsorted(cdf, q, side='left')), len(errors) - 1)
        shifts.append(max(0., float(sorted_errors[index])))
    return list(dict.fromkeys(shifts))


def choose_setting(names, metric, cal, baseline, rmse_ratio, over_ratio):
    feasible = []
    for position, sid in enumerate(names):
        candidate = metric[sid][cal]['summary']
        if (candidate['macro_rmse'] <= rmse_ratio * baseline['macro_rmse'] + 1e-12
                and candidate['macro_over_mae'] <= over_ratio * baseline['macro_over_mae'] + 1e-12):
            feasible.append((candidate['macro_under_rmse'], candidate['macro_rmse'],
                             candidate['macro_over_mae'], position, sid))
    return min(feasible)[-1] if feasible else None


def reconstruct(a, helper, cfg, inputs, source, components, metadata, threshold):
    roles = [r['role'] for r in cfg['role_inputs']]
    ids = {r['role']: r['ids'] for r in cfg['role_inputs']}
    cal = cfg['calibration_role']
    dt = cfg['metrics']['sample_interval_seconds']
    settings, metric, shift_ids = [], {}, {}

    def profiles(cid, role, shift=0.):
        return [(pid, source[(role, pid)]['truth'], components[cid][role][pid] + shift,
                 source[(role, pid)]['index']) for pid in ids[role]]

    def add(c, sid, family, shift):
        setting = dict(id=sid, family=family, base=c['id'], other=None, weight=None,
                       shift=float(shift), backbone=c['backbone'], seed=c['seed'], loss=c['loss'])
        settings.append(setting)
        metric[sid] = {}
        for role in roles:
            a.budget()
            metric[sid][role] = helper.group_metrics(profiles(c['id'], role, shift), dt, threshold)

    for c in inputs['components']:
        add(c, c['id'] + '__raw', 'context' if c['backbone'] == 'context' else 'unadjusted', 0.)
    for c in inputs['components']:
        if c['backbone'] == 'context':
            continue
        names = []
        for index, shift in enumerate(shifts_for(profiles(c['id'], cal))):
            sid = c['id'] + '__shift_' + str(index)
            names.append(sid)
            add(c, sid, 'shift', shift)
        shift_ids[c['id']] = names
    return settings, metric, shift_ids


def policies_reference(a, cfg, metadata, metric, shift_ids):
    """One common raw MSE denominator for every loss of each backbone/seed."""
    cal = cfg['calibration_role']
    roles = [r['role'] for r in cfg['role_inputs']]
    expected = {}
    for cid, c in metadata.items():
        if c['backbone'] == 'context':
            continue
        bases = [other for other in metadata.values() if other['backbone'] == c['backbone']
                 and other['seed'] == c['seed'] and other['loss'] == 'mse']
        a.require(len(bases) == 1, 'Exactly one matched raw MSE budget denominator')
        base_id = bases[0]['id'] + '__raw'
        baseline = metric[base_id][cal]['summary']
        for rr in RMSE_RATIOS:
            for oo in OVER_RATIOS:
                chosen = choose_setting(shift_ids[cid], metric, cal, baseline, rr, oo)
                violations = None
                if chosen is not None:
                    violations = {}
                    for role in roles:
                        selected = metric[chosen][role]['summary']
                        raw_mse = metric[base_id][role]['summary']
                        violations[role] = dict(rmse=selected['macro_rmse'] > rr * raw_mse['macro_rmse'] + 1e-12,
                            over_mae=selected['macro_over_mae'] > oo * raw_mse['macro_over_mae'] + 1e-12)
                key = (c['backbone'], c['seed'], c['loss'], rr, oo)
                expected[key] = dict(backbone=c['backbone'], seed=c['seed'], loss=c['loss'], family='shift',
                    rmse_ratio=rr, over_mae_ratio=oo, base_setting_id=base_id, calibration_role=cal,
                    status='SELECTED' if chosen is not None else 'NO_FEASIBLE_CAL_SETTING',
                    selected_setting_id=chosen, candidate_setting_ids=shift_ids[cid],
                    feasible_setting_ids=[sid for sid in shift_ids[cid]
                        if metric[sid][cal]['summary']['macro_rmse'] <= rr * baseline['macro_rmse'] + 1e-12
                        and metric[sid][cal]['summary']['macro_over_mae'] <= oo * baseline['macro_over_mae'] + 1e-12],
                    evaluation_budget_violations=violations)
    return expected


def policy_key(value):
    return value['backbone'], value['seed'], value['loss'], value['rmse_ratio'], value['over_mae_ratio']


def verify_policies(a, saved, expected, selection_only=False):
    keys = [policy_key(p) for p in saved]
    a.require(len(keys) == len(set(keys)) and set(keys) == set(expected),
              'Complete unique loss/backbone/seed/all16 budget policies')
    for value in saved:
        reference = dict(expected[policy_key(value)])
        if selection_only:
            reference.pop('evaluation_budget_violations')
        a.same(value, reference, 'Independent common-budget CAL selection and null/violations')


def seed_aggregation(expected, metric, seeds, roles):
    groups = []
    for rr in RMSE_RATIOS:
        for oo in OVER_RATIOS:
            for loss in LOSSES:
                for role in roles:
                    rows = []
                    for seed in seeds:
                        policy = expected[('mlp', seed, loss, rr, oo)]
                        sid = policy['selected_setting_id']
                        rows.append(dict(seed=seed, status=policy['status'], selected_setting_id=sid,
                            metrics=None if sid is None else metric[sid][role]['summary'],
                            evaluation_budget_violations=None if sid is None
                            else policy['evaluation_budget_violations'][role]))
                    all_feasible = all(row['metrics'] is not None for row in rows)
                    average = None
                    if all_feasible:
                        names = [k for k in rows[0]['metrics'] if k.startswith('macro_')] + ['pooled_rmse']
                        average = {}
                        for name in names:
                            values = [row['metrics'][name] for row in rows]
                            average[name] = None if any(value is None for value in values) else float(np.mean(values))
                    groups.append(dict(backbone='mlp', loss=loss, rmse_ratio=rr, over_mae_ratio=oo, role=role,
                        expected_seeds=seeds, per_seed=rows,
                        status='ALL_INCLUDED_SEEDS_FEASIBLE' if all_feasible else 'INCOMPLETE_FEASIBILITY',
                        mean_of_seed_metrics=average, three_seed_panel=seeds == SEEDS))
    return dict(seed_scope='THREE_PAIRED_SEEDS' if seeds == SEEDS else 'INITIAL_SEED_IMPLEMENTATION_CHECK',
        seeds=seeds, aggregation='arithmetic mean of per-seed metrics; no ensemble, no feasible-seed-only mean', groups=groups)


def audit_stage(a, helper, args, folder):
    protocol = dict(path='followup-risk-20261110/protocols/risk-calibration-symmetry-v002.md', sha256=PROTOCOL_SHA)
    metric_source = dict(path='followup-risk-20261110/code/risk_screen_analysis_v001.py', sha256=METRICS_PRODUCER_SHA)
    a.pin(protocol)
    a.pin(metric_source)
    a.pin(dict(path='followup-risk-20261110/code/risk_audit_v001.py', sha256=SHARED_SHA))
    a.pin(dict(path='followup-risk-20261110/code/risk_calibration_v002.py', sha256=args.producer_sha256))
    cfg = helper.load_config(a, args)
    original = read(a.pin(cfg['original_manifest']))
    a.same(cfg['fit_cache'], original['fit_cache'], 'Unchanged original FIT cache identities')
    a.same([r['role'] for r in cfg['role_inputs']], [r for r in original['roles'] if r != 'fit'],
           'Complete original ordered roles')
    for key, value in dict(shift_grid_c=SHIFT_GRID, shift_residual_quantiles=QUANTILES,
                          rmse_budget_ratios=RMSE_RATIOS, over_mae_budget_ratios=OVER_RATIOS,
                          primary_rmse_ratio=1.05, primary_over_mae_ratio=1.25).items():
        a.same(cfg['calibration'][key], value, 'Original frozen calibration/' + key, atol=0., rtol=0.)
    a.require(not (folder / 'failure.json').exists(), 'No terminal failure in completed v002 stage')
    summary, files = helper.summary_files(a, folder, args.input_sha256)
    required = {'configuration.json', 'component-inputs.json', 'settings.json', 'policy_selections.json',
                'calibration_metrics.json', 'calibration_lock.json', 'metrics.json', 'policies.json',
                'paired_seed_summary.json', 'input_pins.json'}
    a.require(set(files) == required, 'Exact complete v002 analysis artifact set')
    a.require(summary['status'] == 'EXPLORATORY_RISK_CALIBRATION_V002_COMPLETE'
              and summary['dataset'] == cfg['dataset'] and summary['config_sha256'] == args.config_sha256
              and summary['source_sha256'] == args.producer_sha256
              and summary['metric_source_sha256'] == METRICS_PRODUCER_SHA, 'Completed same-source v002 summary')
    a.same(summary['protocol'], protocol, 'Completed stage frozen addendum')
    a.require(summary['independent_confirmation'] is False and summary['new_fit_performed'] is False
              and summary['evaluation_used_for_selection'] is False, 'Exploratory no-refit/CAL-only scope')
    configuration = read(folder / 'configuration.json')
    a.require(configuration['source_sha256'] == args.producer_sha256, 'Configured producer SHA')
    a.same(configuration['protocol'], protocol, 'Configuration protocol pin')
    a.same(configuration['metric_dependency'], metric_source, 'Configuration frozen metric dependency')
    manifest_ref = dict(path=configuration['components'], sha256=configuration['components_sha256'])
    original_inputs = read(a.pin(manifest_ref))
    inputs = read(folder / 'component-inputs.json')
    a.same(inputs, original_inputs, 'Exact source component manifest copy')
    a.require(inputs['dataset'] == cfg['dataset'] and inputs['config']['sha256'] == args.config_sha256
              and a.pin(inputs['config']) == local(args.config), 'Original source/config identity')
    lock = read(folder / 'calibration_lock.json')
    a.require(lock['schema'] == 'c16-risk-calibration-selection-lock-v002'
              and lock['status'] == 'CAL_ONLY_SELECTIONS_LOCKED' and lock['dataset'] == cfg['dataset']
              and lock['source_sha256'] == args.producer_sha256 and lock['evaluation_truth_accessed'] is False,
              'Same-source CAL-only choice lock')
    a.same(lock['protocol'], protocol, 'Calibration lock protocol')
    a.same(lock['components'], manifest_ref, 'Calibration lock exact input manifest')
    a.same(lock['config'], inputs['config'], 'Calibration lock original dataset config')
    a.same(lock['files'], {name: files[name] for name in
                          ('settings.json', 'policy_selections.json', 'calibration_metrics.json')},
           'Complete immutable CAL selection artifacts')
    all_pins = read(folder / 'input_pins.json')
    for name, digest in all_pins.items():
        a.pin(dict(path=name, sha256=digest))
    before = lock['input_hashes_before_evaluation_truth']
    for name, digest in before.items():
        a.require(all_pins.get(name) == digest, 'Before-evaluation input identity preserved')
    held_truth_paths = {local(entry['truth']['path']) for role in cfg['role_inputs']
                        if role['role'] != cfg['calibration_role'] for entry in role['records']}
    a.require(not held_truth_paths.intersection(local(name) for name in before),
              'No held target NPZ in pre-selection input receipt')
    for role in cfg['role_inputs']:
        for entry in role['records']:
            path = str(local(entry['truth']['path']))
            a.require(all_pins.get(path) == entry['truth']['sha256'], 'Final receipt binds every reported target')
            if role['role'] == cfg['calibration_role']:
                a.require(before.get(path) == entry['truth']['sha256'], 'CAL target bound before choices')
    threshold = helper.high_threshold_only(a, cfg)
    a.same(summary['high_temperature_threshold_c'], threshold, 'FIT-only high-temperature threshold')
    source, components, metadata, seeds = load_components(a, helper, cfg, inputs, args.config_sha256)
    a.same(summary['seeds'], seeds, 'Summary complete ordered neural seed panel')
    a.same(lock['seeds'], seeds, 'CAL lock seed panel')
    a.same(lock['calibration_role'], cfg['calibration_role'], 'CAL lock role')
    a.same(summary['complete_three_seed_panel'], seeds == SEEDS, 'First seed cannot be called complete three-seed evidence')
    settings, metric, shift_ids = reconstruct(a, helper, cfg, inputs, source, components, metadata, threshold)
    a.same(read(folder / 'settings.json'), settings, 'All symmetric own-loss shifts and unchanged raw controls')
    a.same(read(folder / 'metrics.json'), dict(dataset=cfg['dataset'], high_temperature_threshold_c=threshold,
                                            settings=metric), 'Every independent risk metric', atol=1e-8, rtol=1e-10)
    cal = cfg['calibration_role']
    a.same(read(folder / 'calibration_metrics.json'), dict(dataset=cfg['dataset'], high_temperature_threshold_c=threshold,
             settings={sid: values[cal] for sid, values in metric.items()}), 'Locked CAL-only metrics', atol=1e-8, rtol=1e-10)
    expected_policies = policies_reference(a, cfg, metadata, metric, shift_ids)
    policies = read(folder / 'policies.json')
    selections = read(folder / 'policy_selections.json')
    verify_policies(a, policies, expected_policies)
    verify_policies(a, selections, expected_policies, selection_only=True)
    a.require([policy_key(row) for row in policies] == [policy_key(row) for row in selections],
              'Selections preserved in final reporting')
    roles = [cal] + [r['role'] for r in cfg['role_inputs'] if r['role'] != cal]
    seed_summary = seed_aggregation(expected_policies, metric, seeds, roles)
    a.same(read(folder / 'paired_seed_summary.json'), seed_summary, 'Paired seeds; null means when any included seed infeasible',
           atol=1e-8, rtol=1e-10)
    a.same(summary['settings'], len(settings), 'Complete candidate count')
    a.same(summary['policies'], len(expected_policies), 'Complete all16 budget policy count')
    no_feasible = sum(row['selected_setting_id'] is None for row in expected_policies.values())
    a.same(summary['no_feasible_policy_count'], no_feasible, 'All infeasible policies retained')
    a.require(not (folder / 'failure.json').exists(), 'Completed producer remains free of failure marker')
    a.results = dict(dataset=cfg['dataset'], settings=len(settings), policies=len(expected_policies),
        no_feasible_policy_count=no_feasible, seeds=seeds, complete_three_seed_panel=seeds == SEEDS,
        method_assessment_requires_all_three_seeds=True, evaluated_roles=roles,
        high_temperature_threshold_c=threshold, independent_confirmation=False,
        all_evaluation_previously_exposed=True,
        primary_budget_policies=[p for p in policies if p['rmse_ratio'] == 1.05 and p['over_mae_ratio'] == 1.25],
        paired_seed_summary=seed_summary)
    a.replayed = dict(settings='Every component own CAL-only profile-balanced residual quantile and fixed shift',
        selection='All16 budgets against common matched raw MSE; exact null and declared-order tie handling',
        metrics='Every candidate/profile/role; immutable independent v001 metric implementation',
        neural='Original independent v001 forward PASS and current prediction/receipt/source hash binding; no new forward pass',
        tree='Saved prediction/hash/index/producer lineage only; no pickle or tree forward pass',
        seed_means='Arithmetic mean of per-seed metrics, never selected feasible-seed subset or prediction ensemble')


def synthetic_self_test():
    """Constructed arithmetic only: null, tie order, own shifts, no partial mean."""
    baseline = dict(macro_rmse=1., macro_over_mae=1., macro_under_rmse=1.)
    metric = {'first': {'cal': {'summary': dict(baseline)}},
              'second': {'cal': {'summary': dict(baseline)}},
              'bad': {'cal': {'summary': dict(macro_rmse=2., macro_over_mae=2., macro_under_rmse=0.)}}}
    assert choose_setting(['bad'], metric, 'cal', baseline, 1., 1.) is None
    assert choose_setting(['second', 'first'], metric, 'cal', baseline, 1., 1.) == 'second'
    profiles = [(1, np.array([20., 20.]), np.zeros(2), np.array([300, 301], dtype=np.int64))]
    shifts = shifts_for(profiles)
    assert shifts == SHIFT_GRID + [20.] and len(shifts) == len(set(shifts))
    negative = [(1, np.array([-20., -20.]), np.zeros(2), np.array([300, 301], dtype=np.int64))]
    assert shifts_for(negative) == SHIFT_GRID
    expected = {}
    complete_metric = {'selected': {'role': {'summary': dict(macro_rmse=2., macro_under_rmse=1.,
                             macro_over_mae=.5, pooled_rmse=3., macro_under_gt5_longest_seconds=None)}}}
    for rr in RMSE_RATIOS:
        for oo in OVER_RATIOS:
            for loss in LOSSES:
                for seed in SEEDS:
                    sid = None if seed == SEEDS[1] and loss == 'asym4' else 'selected'
                    expected[('mlp', seed, loss, rr, oo)] = dict(selected_setting_id=sid,
                        status='SELECTED' if sid else 'NO_FEASIBLE_CAL_SETTING',
                        evaluation_budget_violations={'role': dict(rmse=False, over_mae=False)} if sid else None)
    summary = seed_aggregation(expected, complete_metric, SEEDS, ['role'])
    assert summary['seed_scope'] == 'THREE_PAIRED_SEEDS'
    for group in summary['groups']:
        if group['loss'] == 'asym4':
            assert group['mean_of_seed_metrics'] is None and group['status'] == 'INCOMPLETE_FEASIBILITY'
        else:
            assert group['mean_of_seed_metrics']['macro_rmse'] == 2.
            assert group['mean_of_seed_metrics']['macro_under_gt5_longest_seconds'] is None
    return dict(status='SYNTHETIC_CALIBRATION_RULES_PASS', scope='Constructed arrays/metadata only; no campaign data read')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--config-sha256')
    parser.add_argument('--input-dir')
    parser.add_argument('--input-sha256', help='Completed v002 summary.json SHA')
    parser.add_argument('--producer-sha256')
    parser.add_argument('--output-dir')
    parser.add_argument('--max-seconds', type=float, default=900.)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(synthetic_self_test()))
        return 0
    for key in ('config', 'config_sha256', 'input_dir', 'input_sha256', 'producer_sha256', 'output_dir'):
        if not getattr(args, key):
            parser.error('Required --' + key.replace('_', '-'))
    if not math.isfinite(args.max_seconds) or not 0 < args.max_seconds <= 3600:
        parser.error('Audit cap must be finite in (0,3600]')
    folder, output = local(args.input_dir), local(args.output_dir)
    if not folder.is_relative_to(FOLLOWUP / 'runs') or not output.is_relative_to(FOLLOWUP / 'runs'):
        raise ValueError('New stage inputs/output must remain below followup runs')
    output.mkdir(parents=True, exist_ok=False)
    helper = dependency()
    a = helper.Audit(args.max_seconds)
    error = None
    try:
        audit_stage(a, helper, args, folder)
        a.finish_pins()
    except Exception as exc:
        error = type(exc).__name__ + ': ' + str(exc)
        a.failures.append(dict(check='All independent symmetric-calibration audit contracts', detail=error))
    report = dict(schema='c16-symmetric-calibration-independent-audit-v002',
                  status='HOLD' if a.failures else 'PASS', auditor_sha256=sha(__file__),
                  producer_sha256=args.producer_sha256, config_sha256=args.config_sha256,
                  input_sha256=args.input_sha256, independent_dependency_sha256=SHARED_SHA,
                  checks=a.checks, failures=a.failures, inputs=a.inputs, results=a.results,
                  replayed=a.replayed, error=error, elapsed_seconds=time.monotonic() - a.start,
                  scope='Original v001 forward-audit binding and independent reconstruction of per-loss CAL shifts, '
                        'common-raw-MSE budgets, null selections and every risk metric. No new fit, scientific '
                        'producer import, pickle execution, model-forward replay or fresh independent confirmation.')
    with (output / 'report.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(dict(status=report['status'], checks=a.checks, failures=len(a.failures))))
    return 2 if a.failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
