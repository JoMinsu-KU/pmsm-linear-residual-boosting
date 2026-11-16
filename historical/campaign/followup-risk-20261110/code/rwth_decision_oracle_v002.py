"""Independent, data-loader-free oracle for frozen RWTH result adjudication.

Accepts independently verified metric dictionaries. No producer imports,
model fitting, file input, dataset access, or campaign-state operations.
"""
from __future__ import annotations

import copy
import math

SEEDS = [2026111701, 2026111702, 2026111703]
MLP_METHODS = ['mlp__' + str(seed) for seed in SEEDS]
PAIRS = ['c15', 'direct200', 'linear33', 'free25', 'positive_prior']
PRIMARY = 'c15__cal_macro_rmse'
CONF_IDS = [5, 32, 33, 36, 24, 2, 25, 14, 3, 12]
KEYS = ['rmse', 'under_rmse', 'over_mae']


def average(values):
    numbers = [float(v) for v in values]
    if not numbers or not all(math.isfinite(v) for v in numbers):
        raise ValueError('Nonempty finite metric sequence required')
    return math.fsum(numbers) / len(numbers)


def profile_ids(group):
    ids = [r['id'] for r in group['per_profile']]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Nonempty unique paired profile identities required')
    return ids


def metric_mean(metric, role):
    """Arithmetic mean of seed metrics; never average seed predictions."""
    groups = [metric[method][role] for method in MLP_METHODS]
    ids = profile_ids(groups[0])
    if any(profile_ids(group) != ids for group in groups[1:]):
        raise ValueError('Different neural seed cohorts/order')
    by_profile = []
    for position, pid in enumerate(ids):
        by_profile.append(dict(id=pid, **{
            key: average(g['per_profile'][position][key] for g in groups)
            for key in KEYS}))
    return dict(per_profile=by_profile, summary={
        'macro_' + key: average(g['summary']['macro_' + key] for g in groups)
        for key in KEYS}, seeds=SEEDS.copy(),
        aggregation='arithmetic mean of three seed metrics, not predictions or best seed')


def contrast(candidate, baseline, candidate_name, baseline_name, role):
    ids = profile_ids(candidate)
    if ids != profile_ids(baseline):
        raise ValueError('Different paired metric cohorts/order')
    c, b = candidate['summary'], baseline['summary']
    delta = [float(cr['rmse']) - float(br['rmse'])
             for cr, br in zip(candidate['per_profile'], baseline['per_profile'])]
    if not all(math.isfinite(v) for v in delta):
        raise ValueError('Finite paired differences required')

    def ratio(key):
        numerator, denominator = float(c[key]), float(b[key])
        if not (math.isfinite(numerator) and math.isfinite(denominator)):
            raise ValueError('Finite summary metrics required')
        return numerator / denominator if denominator != 0. else None

    return dict(role=role, candidate=candidate_name, comparator=baseline_name,
        delta_macro_rmse=float(c['macro_rmse']) - float(b['macro_rmse']),
        rmse_ratio=ratio('macro_rmse'), under_rmse_ratio=ratio('macro_under_rmse'),
        over_mae_ratio=ratio('macro_over_mae'), profile_ids=ids,
        per_profile_rmse_differences=delta,
        wins=len([v for v in delta if v < 0.]), ties=len([v for v in delta if v == 0.]),
        losses=len([v for v in delta if v > 0.]),
        under_rmse_diagnostic_flag=c['macro_under_rmse'] > 1.05*b['macro_under_rmse'] + 1e-12,
        over_mae_diagnostic_flag=c['macro_over_mae'] > 1.25*b['macro_over_mae'] + 1e-12,
        risk_flags_scope='Descriptive imported diagnostic ratios, not safety requirements or formal decision criteria')


def exact_sign(delta):
    """Upper tail of a fair binomial in improvement wins, exact integer sum."""
    wins = sum(1 for v in delta if v < 0.)
    losses = sum(1 for v in delta if v > 0.)
    n = wins + losses
    # Generate binomial coefficients by recurrence, independent of math.comb.
    coefficient, numerator = 1, 0
    for j in range(n + 1):
        if j >= wins:
            numerator += coefficient
        if j < n:
            coefficient = coefficient * (n - j) // (j + 1)
    denominator = 1 << n
    return wins, losses, len(delta) - n, n, dict(numerator=numerator,
        denominator=denominator, p_value=numerator/denominator,
        direction='smaller RMSE', zero_ties_excluded=True)


def formal_decision(metrics, mean, models):
    """Frozen protocol sections10/11, same report schema as the producer."""
    role = 'confirmation'
    primary = metrics[PRIMARY][role]
    paired = contrast(primary, metrics['c15__left'][role], PRIMARY, 'c15__left', role)
    if paired['profile_ids'] != CONF_IDS:
        raise ValueError('Exactly the ten predetermined CONF cycles required')
    differences = paired['per_profile_rmse_differences']
    wins, losses, ties, effective_n, sign = exact_sign(differences)
    mean_delta = average(differences)  # Includes all ten cycles, including ties.
    formal = mean_delta < 0. and sign['p_value'] <= .025
    names = [p + '__cal_macro_rmse' for p in PAIRS[1:]] + ['c15__half']
    baselines = [metrics[name][role] for name in names] + [mean]
    names += ['mlp_three_seed_metric_mean']
    comparisons = [contrast(primary, group, PRIMARY, name, role)
                   for name, group in zip(names, baselines)]
    strict = [dict(comparator=name, primary_macro_rmse=primary['summary']['macro_rmse'],
        comparator_macro_rmse=group['summary']['macro_rmse'],
        strictly_lower=primary['summary']['macro_rmse'] < group['summary']['macro_rmse'])
        for name, group in zip(names, baselines)]
    contribution = all(item['strictly_lower'] for item in strict)
    passed = formal and contribution
    weight = models['pairs']['c15']['final']['cal_macro_rmse']['weight']
    if not math.isfinite(weight) or not 0. <= weight <= 1.:
        raise ValueError('A locked CAL convex weight is required')
    verdict = ('FORMAL_AND_CONTRIBUTION_CRITERIA_MET' if passed else
               'FORMAL_PASS_BUT_CONTRIBUTION_NOT_ESTABLISHED' if formal else
               'PRIMARY_CONFIRMATION_FAILED')
    return dict(schema='rwth-confirmation-decision-v002', candidate='C17',
        confirmation_id='C17_RWTH_FIXED_CONFIRMATION_02', primary=PRIMARY,
        formal_comparator='c15__left', alpha=.025, profile_ids=CONF_IDS.copy(),
        mean_delta=mean_delta, per_profile_rmse_differences=differences,
        wins=wins, ties=ties, losses=losses, n_eff=effective_n,
        exact_one_sided_sign=sign, formal_success=formal,
        strict_point_mean_controls=strict, contribution_mean_condition=contribution,
        topic_criterion_pass=passed, verdict=verdict, primary_cal_weight=weight,
        primary_collapses_to='direct9' if weight == 0. else 'C15' if weight == 1. else None,
        leave_one_cycle_out_mean_deltas=[dict(excluded_profile=pid,
            mean_delta=average(v for j, v in enumerate(differences) if j != i))
            for i, pid in enumerate(CONF_IDS)], required_comparisons=comparisons,
        pending_independent_audit=True,
        scope='One additional motor,10 fixed cycles; cycle independence unverified; no secondary significance or safety guarantee')


def synthetic_test():
    """Only fabricated small dictionaries; does not read any dataset/file."""
    def group(values):
        rows = [dict(id=pid, rmse=float(v), under_rmse=1., over_mae=1.)
                for pid, v in zip(CONF_IDS, values)]
        return dict(per_profile=rows, summary={
            'macro_' + key: average(row[key] for row in rows) for key in KEYS})

    def case(differences):
        metric = {PRIMARY: {'confirmation': group([3.+d for d in differences])},
                  'c15__left': {'confirmation': group([3.]*10)}}
        for name in [p+'__cal_macro_rmse' for p in PAIRS[1:]] + ['c15__half'] + MLP_METHODS:
            metric[name] = {'confirmation': group([10.]*10)}
        model = dict(pairs=dict(c15=dict(final=dict(cal_macro_rmse=dict(weight=.25)))))
        return metric, model

    checks = []
    for name, differences, expected in [
        ('all_ties', [0.]*10, (1, 1, False)),
        ('nine_wins', [-.2]*9+[.1], (11, 1024, True)),
        ('positive_mean', [-.1]*9+[2.], (11, 1024, False))]:
        metric, model = case(differences)
        result = formal_decision(metric, metric_mean(metric, 'confirmation'), model)
        sign = result['exact_one_sided_sign']
        assert (sign['numerator'], sign['denominator'], result['formal_success']) == expected
        checks.append(name)
    metric, model = case([-.2]*10)
    metric['direct200__cal_macro_rmse'] = copy.deepcopy(metric[PRIMARY])
    result = formal_decision(metric, metric_mean(metric, 'confirmation'), model)
    assert result['formal_success'] and not result['contribution_mean_condition']
    assert not result['topic_criterion_pass']  # Equality is not strict superiority.
    checks.append('auxiliary_tie_blocks_contribution')
    assert exact_sign([-.1]*8+[0., 0.])[4]['p_value'] == 1/256
    checks.append('ties_excluded_sign_only')
    return dict(status='PASS_SYNTHETIC_ONLY', fixtures=checks, no_dataset_access=True)


if __name__ == '__main__':
    import json
    print(json.dumps(synthetic_test()))
