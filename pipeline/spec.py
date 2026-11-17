from dataclasses import dataclass
from typing import List, Optional, Tuple
LMM = 'pooled MixedLM random-intercept'
OLS = 'per-participant OLS slope, subject-cluster BCa'

@dataclass(frozen=True)
class Model:
    id: str
    measure: str
    predictor: str
    channel: str
    outcome: str = 'p300_pz'
    covariates: Tuple[str, ...] = ()
    estimator: str = LMM
    paper: Optional[str] = None
    alias: Tuple[str, ...] = ()
    note: str = ''

    @property
    def family(self):
        return self.id.rsplit('-', 1)[1]

    @property
    def label(self):
        return self.id.rsplit('-', 1)[0]
M = {'rms_fz': ('RMS Fz 0-150', 'rms_early_fz', 'Fz->Pz'), 'mean_fz': ('mean Fz 0-150', 'mean_early_fz', 'Fz->Pz'), 'sd_fz': ('SD Fz 0-150', 'sd_early_fz', 'Fz->Pz'), 'rms_fz200': ('RMS Fz 0-200', 'rms_early_fz_200', 'Fz->Pz'), 'rms_fz250': ('RMS Fz 0-250', 'rms_early_fz_250', 'Fz->Pz'), 'mean_fz200': ('mean Fz 0-200', 'mean_early_fz_200', 'Fz->Pz'), 'mean_fz250': ('mean Fz 0-250', 'mean_early_fz_250', 'Fz->Pz'), 'rms_base_fz': ('RMS Fz baseline', 'rms_base_fz', 'Fz->Pz'), 'mean_base_fz': ('mean Fz baseline', 'mean_base_fz', 'Fz->Pz'), 'rms_pz': ('RMS Pz 0-150', 'rms_early_pz', 'Pz->Pz'), 'ms_pz': ('mean-square energy Pz', 'ms_early_pz', 'Pz->Pz'), 'mean_pz': ('mean Pz 0-150', 'mean_early_pz', 'Pz->Pz'), 'sd_pz': ('SD Pz 0-150', 'sd_early_pz', 'Pz->Pz'), 'rms_base_pz': ('RMS Pz baseline', 'rms_base_pz', 'Pz->Pz'), 'median_fz': ('median Fz 0-150', 'median_amp_fz', 'Fz->Pz'), 'mad_fz': ('MAD Fz 0-150', 'mad_amp_fz', 'Fz->Pz'), 'p2p_fz': ('peak-to-peak Fz 0-150', 'p2p_amp_fz', 'Fz->Pz'), 'skew_fz': ('skewness Fz 0-150', 'skew_amp_fz', 'Fz->Pz'), 'kurt_fz': ('kurtosis Fz 0-150', 'kurt_amp_fz', 'Fz->Pz'), 'slope_fz': ('slope Fz 0-150', 'slope_amp_fz', 'Fz->Pz'), 'pe_fz': ('permutation entropy Fz', 'pe_early_fz', 'Fz->Pz'), 'se_fz': ('sample entropy Fz', 'se_early_fz', 'Fz->Pz'), 'lz_fz': ('Lempel-Ziv Fz', 'lz_early_fz', 'Fz->Pz'), 'hjmob_fz': ('Hjorth mobility Fz', 'hjmob_early_fz', 'Fz->Pz'), 'hjcplx_fz': ('Hjorth complexity Fz', 'hjcplx_early_fz', 'Fz->Pz')}
_L = [('M1', 'rms_fz', (), 'M1_RMS_Fz_0_150'), ('M2', 'mean_fz', (), 'M2_mean_Fz_only'), ('M3', 'sd_fz', (), 'M3_SD_Fz_only'), ('M4a', 'mean_fz', ('sd_early_fz',), 'M4a_competitive_Fz_mean'), ('M4b', 'sd_fz', ('mean_early_fz',), 'M4b_competitive_Fz_SD'), ('M5', 'rms_fz200', (), 'M5_RMS_Fz_0_200'), ('M5m', 'mean_fz200', (), 'M5m_mean_Fz_0_200'), ('M6', 'rms_fz250', (), 'M6_RMS_Fz_0_250'), ('M6m', 'mean_fz250', (), 'M6m_mean_Fz_0_250'), ('M7', 'rms_base_fz', (), 'M7_baseline_Fz_RMS'), ('M7m', 'mean_base_fz', (), 'M7m_baseline_Fz_mean'), ('M8', 'rms_pz', (), 'M8_RMS_Pz_0_150'), ('M9a', 'mean_pz', ('sd_early_pz',), 'M9a_competitive_Pz_mean'), ('M9b', 'sd_pz', ('mean_early_pz',), 'M9b_competitive_Pz_SD'), ('M10', 'rms_base_pz', (), 'M10_baseline_Pz_RMS'), ('M11', 'rms_pz', ('rms_base_pz',), 'M11_Pz_RMS_with_baseline_cov'), ('M12', 'mean_pz', ('mean_base_pz',), 'M12_Pz_mean_with_baseline_cov')]
_D = [('M1', 'rms_fz', 'M1_RMS_Fz_0_150'), ('M3', 'sd_fz', 'M3_Fz_sd_0_150'), ('M4a', 'mean_fz', 'M4a_mean_Fz'), ('M5', 'rms_fz200', 'M5_Fz_RMS_0_200'), ('M6', 'rms_fz250', 'M6_Fz_RMS_0_250'), ('M7', 'rms_base_fz', 'M7_Fz_baseline_RMS'), ('M8', 'rms_pz', 'M8_RMS_Pz_0_150'), ('M9a', 'mean_pz', 'M9a_mean_Pz'), ('M10', 'rms_base_pz', 'M10_Pz_baseline_RMS'), ('M13', 'pe_fz', 'M13_PE_Fz_0_150'), ('M14', 'se_fz', 'M14_SE_Fz_0_150'), ('M15', 'lz_fz', 'M15_LZ_Fz_0_150'), ('M16', 'skew_fz', 'M_SKEW_Fz_0_150'), ('M17', 'kurt_fz', 'M_KURT_Fz_0_150'), ('M18', 'median_fz', 'M_MED_Fz_0_150'), ('M19', 'mad_fz', 'M_MAD_Fz_0_150'), ('M20', 'p2p_fz', 'M_P2P_Fz_0_150'), ('M21', 'slope_fz', 'M_SLOPE_Fz_0_150'), ('M22', 'hjmob_fz', 'M22_HJORTHMOB_Fz_0_150'), ('M23', 'hjcplx_fz', 'M23_HJORTHCPLX_Fz_0_150'), ('M24', 'ms_pz', '')]
_NOTCOMP = 'single mean predictor without the SD covariate; the competitive model with the SD covariate is the pooled model with the same label (-L).'
_OURS = 'label assigned by this registry.'
_NEW = 'ms_early_pz = rms_early_pz**2, reported beside RMS.'
CANONICAL: List[Model] = []
for lab, k, cov, lg in _L:
    nm, pred, ch = M[k]
    CANONICAL.append(Model(f'{lab}-L', nm, pred, ch, covariates=cov + ('condition',), estimator=LMM, paper=lab, alias=(lg,), note='competitive pair' if cov else ''))
MATCHED: List[Model] = []
for lab, k, lg in _D:
    nm, pred, ch = M[k]
    note = _NOTCOMP if lab in ('M4a', 'M9a') else _OURS if lab in ('M19', 'M21') else _NEW if lab == 'M24' else ''
    MATCHED.append(Model(f'{lab}-D', nm, pred, ch, estimator=OLS, paper=None if lab == 'M24' else lab, alias=(lg,) if lg else (), note=note))
ALL = CANONICAL + MATCHED
_P = [('P1', 'par_tasym_fz', 'time asymmetry Fz 0-150 (residual)', 'Fz->Pz', 'odd'),
      ('P2', 'par_updown_fz', 'up-down asymmetry Fz 0-150 (residual)', 'Fz->Pz', 'odd'),
      ('P3', 'par_klasym_fz', 'directed-KL asymmetry Fz 0-150 (residual)', 'Fz->Pz', 'odd'),
      ('P4', 'par_slope_fz', 'slope Fz 0-150 (residual)', 'Fz->Pz', 'odd'),
      ('E1', 'par_dkurt_fz', 'difference kurtosis Fz 0-150 (residual)', 'Fz->Pz', 'even partner of P1'),
      ('E2', 'par_mono_fz', 'monotone-pattern share Fz 0-150 (residual)', 'Fz->Pz', 'even partner of P2'),
      ('E3', 'par_klsym_fz', 'symmetric KL Fz 0-150 (residual)', 'Fz->Pz', 'even partner of P3'),
      ('B1', 'par_tasym_bfz', 'time asymmetry Fz baseline (residual)', 'Fz->Pz', 'odd, pre-stimulus control'),
      ('B2', 'par_updown_bfz', 'up-down asymmetry Fz baseline (residual)', 'Fz->Pz', 'odd, pre-stimulus control'),
      ('B3', 'par_klasym_bfz', 'directed-KL asymmetry Fz baseline (residual)', 'Fz->Pz', 'odd, pre-stimulus control'),
      ('B4', 'par_slope_bfz', 'slope Fz baseline (residual)', 'Fz->Pz', 'odd, pre-stimulus control'),
      ('Z1', 'par_tasym_pz', 'time asymmetry Pz 0-150 (residual)', 'Pz->Pz', 'odd, same channel'),
      ('Z2', 'par_updown_pz', 'up-down asymmetry Pz 0-150 (residual)', 'Pz->Pz', 'odd, same channel'),
      ('Z3', 'par_klasym_pz', 'directed-KL asymmetry Pz 0-150 (residual)', 'Pz->Pz', 'odd, same channel'),
      ('Z4', 'par_slope_pz', 'slope Pz 0-150 (residual)', 'Pz->Pz', 'odd, same channel'),
      ('R1', 'par_tasym_fz_raw', 'time asymmetry Fz 0-150 (raw)', 'Fz->Pz', 'odd, secondary, no residualisation'),
      ('R2', 'par_updown_fz_raw', 'up-down asymmetry Fz 0-150 (raw)', 'Fz->Pz', 'odd, secondary, no residualisation'),
      ('R3', 'par_klasym_fz_raw', 'directed-KL asymmetry Fz 0-150 (raw)', 'Fz->Pz', 'odd, secondary, no residualisation')]
PARITY: List[Model] = [Model(f'{lab}-D', nm, pred, ch, estimator=OLS, note=note) for lab, pred, nm, ch, note in _P]
BY_ID = {m.id: m for m in ALL}
HEADLINE = ['M1-D', 'M4a-D', 'M8-D', 'M9a-D', 'M13-D', 'M14-D', 'M15-D', 'M22-D', 'M23-D']

def validate():
    pids = [m.id for m in PARITY]
    assert len(set(pids)) == len(pids), 'duplicate parity ids'
    assert not set(pids) & {m.id for m in ALL}, 'parity ids collide with the battery'
    ids = [m.id for m in ALL]
    dup = {i for i in ids if ids.count(i) > 1}
    assert not dup, f'duplicate ids: {dup}'
    for m in ALL:
        assert m.predictor and m.measure and m.channel, f'incomplete: {m.id}'
        assert m.id.endswith(('-L', '-D')), f'{m.id} has no family suffix'
        if m.family == 'L':
            assert 'condition' in m.covariates, f'{m.id} must carry condition'
            assert m.estimator == LMM
        else:
            assert m.covariates == (), f'{m.id} takes no covariates'
            assert m.estimator == OLS
        if m.paper:
            assert m.label == m.paper, f'{m.id}: stripping the suffix gives {m.label!r}, the paper label is {m.paper!r}'
    return len(ALL)

def alias_map(family):
    fam = {'L': 'L', 'D': 'D', 'CL': 'L', 'DB': 'D', 'canonical': 'L', 'matched': 'D'}[family]
    out = {}
    for m in ALL:
        if m.family != fam:
            continue
        for lg in m.alias:
            base = lg.split(' ')[0].strip()
            if base:
                if base in out:
                    raise AssertionError(f'alias {base!r} maps to both {out[base]} and {m.id}')
                out[base] = m.id
    return out

def both_families():
    L = {m.label for m in CANONICAL}
    D = {m.label for m in MATCHED}
    return sorted(L & D)
if __name__ == '__main__':
    print(f'{validate()} models: {len(CANONICAL)} canonical (-L) + {len(MATCHED)} matched (-D)')
    print(f'\nstripping the suffix gives the paper label; {len(both_families())} labels are estimated by both families:')
    print(f"  {', '.join((f'{l}-L / {l}-D' for l in both_families()))}")
    print(f"\ncanonical: {', '.join((m.id for m in CANONICAL))}")
    print(f"matched:   {', '.join((m.id for m in MATCHED))}")
    ours = [m.id for m in ALL if m.note.startswith('label assigned')]
    print(f'\nlabels assigned by this registry: {ours}')
    print(f'models without a paper label: {[m.id for m in ALL if m.paper is None]}')
