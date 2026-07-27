"""Сборка отчёта по одному чекпоинту.

Один вызов — один JSON и один markdown. Все числа идут с доверительными
интервалами: число без интервала сравнивать между прогонами нельзя, это
установлено измерением (epoch-to-epoch разброс полевого recall ~9 пп при
монотонно растущем auc_hard).
"""

import os
import json
import numpy as np

from airadar.bench import corpus, decision, ladder, transfer, strata
from airadar.bench.ci import block_bootstrap, ci

FA_BUDGET = 1.0          # тревог в час: оператор терпит одну, десять — выключит
NOMINAL_FAR = 0.01
OFF_DELTA = 1.0


def run_bench(scorer, name, seed=0):
    rep = {"name": name, "hop_s": scorer.hop_s, "context_s": scorer.context_s}

    # 1. непрерывный трудный фон: рабочая точка и FA/час.
    #    Сначала regroup восстанавливает исходные клипы (окна одной группы
    #    смежны встык), и только потом клипы сшиваются кроссфейдом. Склейка
    #    напрямую из 0.5-секундных окон дала бы стык каждые 0.45 с, и при
    #    контексте 0.5 с его задевало бы каждое окно.
    # Фильтр по трудным категориям обязателен. В cache_hard 58 категорий, а
    # трудных (механический гул и погода) — 16, список лежит в meta["hard"].
    # Без фильтра рабочая точка считалась бы по пулу, где собственно моторных
    # и винтовых звуков единицы процентов, а остальное — лай, плач и стройка.
    # Такое среднее насыщается по построению, и именно на это указывает
    # docs/metrics-plan.md §0.5.
    hard, cats, grp = corpus.hard_holdout(cat_filter=corpus.hard_categories())
    clips = corpus.regroup(hard, grp)
    track, seams = corpus.stitch(clips)
    mask = corpus.seam_mask(_n(scorer, len(track)), seams,
                            scorer.context_s, scorer.hop_s)
    if mask.sum() < 0.2 * len(mask):
        raise RuntimeError(
            f"стыки съели {100*(1-mask.mean()):.0f}% окон при контексте "
            f"{scorer.context_s} с — нужен непрерывный корпус из исходников "
            f"(этап 4), а не склейка нарезанного кэша")
    lg_bg = decision.smooth(scorer.score(track), scorer.hop_s)
    on = decision.threshold_for_fa(lg_bg, scorer.hop_s, FA_BUDGET,
                                   OFF_DELTA, mask=mask)
    rep["operating_point"] = {
        "fa_budget_per_hour": FA_BUDGET, "on": on, "off": on - OFF_DELTA,
        "fa_actual": decision.fa_per_hour(lg_bg, scorer.hop_s, on,
                                          on - OFF_DELTA, mask=mask),
        "background_hours": float(len(track) / corpus.SR / 3600.0),
    }

    # 2. лестница SNR50 по каждой полевой записи отдельно.
    #    Усреднять по записям нельзя: у них разная основная частота, и
    #    среднее спрятало бы, что одна пропускается целиком.
    field = corpus.field_records()          # читаем один раз, используем в §2 и §3
    pool = [clips[i] for i in np.linspace(0, len(clips) - 1, 64).astype(int)]
    rep["snr50"] = {}
    for nm, audio in field.items():
        curve = ladder.p_detect_curve(scorer, audio, pool, on, on - OFF_DELTA,
                                      seed=seed)
        rep["snr50"][nm] = {
            "curve": [float(v) for v in curve],
            "snrs": [float(v) for v in ladder.SNR_GRID],
            "snr50_db": ladder.snr50(ladder.SNR_GRID, curve),
        }

    # 3. auc_fh и медианный перцентиль с блочным CI
    lg_hard = scorer.score(track)[mask]
    rep["field"] = {}
    for nm, audio in field.items():
        lg_f = scorer.score(audio)
        rep["field"][nm] = {
            "n_windows": int(len(lg_f)),
            "auc_fh": _auc(lg_f, lg_hard),
            "auc_fh_ci": ci(block_bootstrap(lg_f, lambda v: _auc(v, lg_hard),
                                            n_boot=400, block=12, seed=seed)),
            "median_pct": float(np.mean(lg_hard[None, :] < np.median(lg_f))),
        }

    # 4. перенос порога: фон DADS (лёгкий, лабораторный) -> трудные негативы
    Xd, yd, spd, _cd, gd = corpus.load_cache("cache_dads")
    sel = np.sort(np.flatnonzero((yd == 0) & (spd != 0))[:20000])
    Xn = np.ascontiguousarray(Xd[sel]).astype(np.float32) / 32768.0
    lg_dads = np.concatenate([scorer.score(t)
                              for t in corpus.regroup(Xn, gd[sel])])
    rep["transfer"] = transfer.transfer_error(lg_dads, lg_hard, NOMINAL_FAR)
    rep["transfer"]["drift_p99"] = transfer.drift(lg_dads, lg_hard)

    # 5. recall по f0-полосам, отчётная величина — худшая полоса.
    #    Порог берётся из рабочей точки FA/час, а не из перцентиля лёгких
    #    позитивов: порог, откалиброванный по квадрокоптеру с близкой
    #    дистанции, к тяжёлому дрону отношения не имеет.
    try:
        idx, f0 = strata.load_f0_estimates()
    except (FileNotFoundError, KeyError) as e:
        rep["strata"] = {"error": str(e)}
    else:
        keep = np.isin(idx, np.flatnonzero((yd == 1) & (spd != 0)))
        idx, f0 = idx[keep], f0[keep]
        order = np.argsort(idx)
        idx, f0 = idx[order], f0[order]
        Xp = np.ascontiguousarray(Xd[idx]).astype(np.float32) / 32768.0
        lg_pos = score_windows(scorer, Xp)
        rec = strata.recall_by_band(lg_pos, np.ones(len(lg_pos), int), f0, on)
        name, val = strata.worst_band(rec)
        rep["strata"] = {"by_band": rec, "worst_band": name, "worst_recall": val,
                         "n_windows": int(len(lg_pos))}
    return rep


def _n(scorer, n_samples):
    from airadar.bench.scorer import n_scores
    return n_scores(n_samples, scorer.context_s, scorer.hop_s)


def score_windows(scorer, X):
    """По одной оценке на изолированное окно [N, win].

    Если контекст скорера длиннее окна, окно зацикливается до контекста.
    Это искусственно, и для f0-страт допустимо только потому, что все модели
    получают одну и ту же обработку, а сравниваются они между собой. Для
    будущего 4-секундного скорера страту надо будет пересобрать на клипах —
    отмечено в спецификации как ограничение этапа 0.
    """
    win = X.shape[1]
    ctx = int(round(scorer.context_s * corpus.SR))
    out = np.empty(len(X), np.float32)
    for i, w in enumerate(X):
        a = w if win >= ctx else np.tile(w, int(np.ceil(ctx / win)))[:ctx]
        s = scorer.score(a.astype(np.float32))
        out[i] = s[len(s) // 2]
    return out


def _auc(pos, neg):
    from sklearn.metrics import roc_auc_score
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return float(roc_auc_score(y, np.r_[pos, neg]))


def write_report(rep, out_dir="bench_out"):
    os.makedirs(out_dir, exist_ok=True)
    jp = os.path.join(out_dir, f"{rep['name']}.json")
    mp = os.path.join(out_dir, f"{rep['name']}.md")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    with open(mp, "w", encoding="utf-8") as f:
        f.write(_markdown(rep))
    return jp, mp


def _markdown(rep):
    op = rep["operating_point"]
    out = [f"# {rep['name']}", "",
           f"Рабочая точка: бюджет {op['fa_budget_per_hour']} тревог/час, "
           f"фактически {op['fa_actual']:.2f} на {op['background_hours']:.2f} ч фона.",
           f"Порог включения {op['on']:.3f}, выключения {op['off']:.3f}.", "",
           "| запись | SNR50, дБ | auc_fh | 95% CI auc_fh |", "|---|---|---|---|"]
    for nm in rep["snr50"]:
        s = rep["snr50"][nm]["snr50_db"]
        f_ = rep["field"][nm]
        lo, hi = f_["auc_fh_ci"]
        out.append(f"| {nm} | {s:.1f} | {f_['auc_fh']:.3f} | [{lo:.3f}, {hi:.3f}] |")
    t = rep["transfer"]
    out += ["", f"Перенос порога DADS→трудные: номинал {t['far_nominal']:.3f}, "
                f"факт {t['far_actual']:.3f}, отношение **{t['ratio']:.1f}×**, "
                f"дрейф p99 {t['drift_p99']:.2f} σ."]
    st = rep.get("strata", {})
    if "by_band" in st:
        out += ["", "| f0-полоса, Гц | recall |", "|---|---|"]
        out += [f"| {k} | {v:.3f} |" for k, v in sorted(st["by_band"].items())]
        out += ["", f"Худшая полоса: **{st['worst_band']} Гц, "
                    f"recall {st['worst_recall']:.3f}** "
                    f"({st['n_windows']} окон). Это и есть отчётная величина — "
                    f"среднее по полосам скрывает именно тяжёлые машины.",
                 "", "Recall по полосам считается на сыром (несглаженном) "
                    "логите изолированного окна против того же порога `on`, "
                    "что и рабочая точка — но `on` откалиброван по "
                    "сглаженному EMA-потоку (`decision.smooth`), который "
                    "гасит одиночные всплески. Сырому мгновенному пику "
                    "проще пробить этот порог, чем сглаженному значению в "
                    "потоке, поэтому recall по полосам НЕ сравним напрямую "
                    "с `fa_actual`/`snr50` выше — те посчитаны на "
                    "сглаженном+гистерезисном пайплайне, а это число — нет."]
    else:
        out += ["", f"Страты не посчитаны: {st.get('error', 'нет данных')}"]
    return "\n".join(out) + "\n"
