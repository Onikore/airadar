"""Сборка отчёта по одному чекпоинту.

Один вызов — один JSON и один markdown. Все числа идут с доверительными
интервалами: число без интервала сравнивать между прогонами нельзя, это
установлено измерением (epoch-to-epoch разброс полевого recall ~9 пп при
монотонно растущем auc_hard).

Второе правило, равное первому: величина, которую корпус не в состоянии
разрешить, отчитывается как НЕОПРЕДЕЛЁННАЯ, а не как удачно выполненный
бюджет. Рабочая точка по FA/час требует фона, на котором ожидаемое число
событий при бюджете исчисляется единицами; на 0.3 часа склеенного кэша такого
фона нет, и молчаливый запасной порог здесь — ровно тот способ сравнить шум,
ради прекращения которого харнес и написан.
"""

import os
import json
import math
import subprocess

import numpy as np

from airadar.bench import corpus, decision, ladder, transfer, strata
from airadar.bench.ci import block_bootstrap, ci
from airadar.bench.scorer import check_scorer, n_scores

FA_BUDGET = 1.0          # тревог в час: оператор терпит одну, десять — выключит
NOMINAL_FAR = 0.01
OFF_DELTA = 1.0
TAU_S = 2.0              # постоянная сглаживания решающего слоя

# Сколько событий должно ожидаться на фоне при заданном бюджете, чтобы бюджет
# вообще можно было по этому фону откалибровать. FA/час на корпусе длиной H
# часов квантована шагом 1/H, а число событий пуассоново: при ожидаемых k
# событиях относительная неопределённость ~1/sqrt(k). k=5 даёт ~45% — это уже
# грубо, но ещё осмысленно; при k<1 (наш случай: 0.29 ч даёт k=0.29) выполнить
# бюджет можно ТОЛЬКО нулём событий, а ноль недостижим любым порогом из ряда,
# и подбор вырождается в «максимум фона плюс off_delta».
MIN_EXPECTED_EVENTS = 5.0

# Повторов на ступень лестницы. При n_rep=8 биномиальная ошибка одной точки
# кривой около p=0.5 равна 0.18, и одно это размывает пересечение до ±3 дБ —
# половина той самой единицы «6 дБ ≈ вдвое по дальности», в которой метрика
# измеряется. 32 повтора дают 0.088, то есть ±1.5 дБ от одной точки; цена —
# 15 ступеней × 32 × 2 записи = 960 вызовов scorer.score. Бутстрап поверх
# ничего не стоит: он ресэмплирует уже готовую матрицу исходов.
N_REP = 32
N_BOOT_SNR50 = 1000


def run_bench(scorer, name, seed=0, model_path=None):
    # Контракт скорера проверяется ДО первой метрики. Рассинхронизация длины
    # ряда и шага сдвигает всю разметку по времени и выглядит потом как
    # «модель хуже», а не как баг; проверка стоит один вызов на 4 с тишины.
    check_scorer(scorer)

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
    hard_cats = [str(c) for c in corpus.hard_categories()]
    hard, cats, grp = corpus.hard_holdout(cat_filter=corpus.hard_categories())
    clips = corpus.regroup(hard, grp)
    track, seams = corpus.stitch(clips)
    n_win = n_scores(len(track), scorer.context_s, scorer.hop_s)

    # Две маски, и это не дублирование. mask_raw — окна, чей собственный
    # отрезок не задевает стык; она годится для метрик на СЫРОМ логите.
    # mask_sm дополнительно снимает окна, куда EMA протащила значения из-за
    # стыка: рабочая точка считается на сглаженном ряде, и на нём гарантия
    # mask_raw не действует вовсе.
    mask_raw = corpus.seam_mask(n_win, seams, scorer.context_s, scorer.hop_s)
    mask_sm = corpus.seam_mask_smoothed(n_win, seams, scorer.context_s,
                                        scorer.hop_s, TAU_S)
    if mask_raw.sum() < 0.2 * len(mask_raw):
        raise RuntimeError(
            f"стыки съели {100*(1-mask_raw.mean()):.0f}% окон при контексте "
            f"{scorer.context_s} с — нужен непрерывный корпус из исходников "
            f"(этап 4), а не склейка нарезанного кэша")

    lg_raw = scorer.score(track)              # считается один раз на весь отчёт
    lg_bg = decision.smooth(lg_raw, scorer.hop_s, TAU_S)

    hours_total = len(track) / corpus.SR / 3600.0
    hours_scored = float(mask_sm.sum()) * scorer.hop_s / 3600.0
    expected_events = hours_scored * FA_BUDGET

    op = {
        "fa_budget_per_hour": FA_BUDGET,
        "tau_s": TAU_S,
        # два разных «часа», раньше отчитывавшихся под одним именем:
        # total — вся склеенная дорожка, scored — то, что реально попало в
        # знаменатель FA/час после вычета стыков и памяти EMA
        "background_hours_total": float(hours_total),
        "background_hours_scored": hours_scored,
        "seam_mask_kept_raw": float(mask_raw.mean()),
        "seam_mask_kept_smoothed": float(mask_sm.mean()),
        "fa_resolution_per_hour": decision.fa_resolution_per_hour(
            int(mask_sm.sum()), scorer.hop_s),
        "expected_events_at_budget": float(expected_events),
        "min_expected_events": MIN_EXPECTED_EVENTS,
    }

    if mask_sm.sum() < 0.2 * len(mask_sm):
        op.update(threshold_calibrated=False, threshold_source="seam_mask_too_thin",
                  on=None, off=None, fa_actual=None,
                  reason=(f"после вычета памяти сглаживания (tau={TAU_S} с) от "
                          f"стыков осталось {100*mask_sm.mean():.0f}% окон — "
                          f"недостаточно фона для честного FA/час, нужен "
                          f"непрерывный корпус из исходников (этап 4 плана)"))
    elif expected_events < MIN_EXPECTED_EVENTS:
        op.update(threshold_calibrated=False, threshold_source="unresolvable_corpus",
                  on=None, off=None, fa_actual=None,
                  reason=(f"недостаточно фона для честного FA/час при "
                          f"tau={TAU_S} с: {hours_scored:.3f} ч зачтённого фона "
                          f"при бюджете {FA_BUDGET} тревог/час дают ожидаемо "
                          f"{expected_events:.2f} события, нужно не меньше "
                          f"{MIN_EXPECTED_EVENTS:.0f}. Уложиться в бюджет можно "
                          f"только нулём событий, а ноль недостижим любым "
                          f"порогом из ряда — нужен непрерывный корпус из "
                          f"исходников (этап 4 плана)"))
    else:
        thr = decision.threshold_for_fa(lg_bg, scorer.hop_s, FA_BUDGET,
                                        OFF_DELTA, mask=mask_sm)
        if not thr.found:
            op.update(threshold_calibrated=False, threshold_source="grid_exhausted",
                      on=None, off=None, fa_actual=None,
                      reason=("сетка порогов исчерпана: бюджет не достигается "
                              "ни одним порогом из самого ряда, вернулось бы "
                              "запасное значение max(ряд)+off_delta"))
        else:
            op.update(threshold_calibrated=True, threshold_source="grid_search",
                      on=thr.on, off=thr.on - OFF_DELTA,
                      fa_actual=decision.fa_per_hour(lg_bg, scorer.hop_s, thr.on,
                                                     thr.on - OFF_DELTA,
                                                     mask=mask_sm))
    rep["operating_point"] = op
    on = op["on"]
    off = op["off"]

    field = corpus.field_records()          # читаем один раз, используем в §2 и §3

    # 2. лестница SNR50 по каждой полевой записи отдельно.
    #    Усреднять по записям нельзя: у них разная основная частота, и
    #    среднее спрятало бы, что одна пропускается целиком.
    #    Лестница целиком стоит на рабочей точке: без порогов on/off нет и
    #    события «обнаружено». Подставить сюда любой другой порог значило бы
    #    измерить не тот детектор, который поедет в поле.
    if not op["threshold_calibrated"]:
        rep["snr50"] = {"defined": False, "reason": op["reason"]}
    else:
        pool = [clips[i] for i in np.linspace(0, len(clips) - 1, 64).astype(int)]
        rep["snr50"] = {"defined": True, "n_rep": N_REP, "n_boot": N_BOOT_SNR50,
                        "by_record": {}}
        for nm, audio in field.items():
            curve, hits = ladder.p_detect_curve(scorer, audio, pool, on, off,
                                                n_rep=N_REP, seed=seed,
                                                tau_s=TAU_S, return_hits=True)
            boot = ladder.bootstrap_curves(hits, n_boot=N_BOOT_SNR50, seed=seed)
            lo, hi = ladder.snr50_ci(ladder.SNR_GRID, boot)
            rep["snr50"]["by_record"][nm] = {
                "curve": [float(v) for v in curve],
                "snrs": [float(v) for v in ladder.SNR_GRID],
                "snr50_db": ladder.snr50(ladder.SNR_GRID, curve),
                "snr50_ci": [lo, hi],
                "n_boot_finite": int(np.isfinite(
                    [ladder.snr50(ladder.SNR_GRID, p) for p in boot]).sum()),
            }

    # 3. auc_fh и медианный перцентиль с блочным CI.
    #    От рабочей точки не зависят: это ранговые величины на сыром логите,
    #    поэтому считаются в любом исходе §1.
    lg_hard = lg_raw[mask_raw]
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

    # 4. перенос порога: фон DADS (лёгкий, лабораторный) -> трудные негативы.
    #    Тоже не зависит от рабочей точки: порог здесь свой, по доле окон.
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
    #    дистанции, к тяжёлому дрону отношения не имеет. Нет рабочей точки —
    #    нет и recall; отчитывается только состав страт, он от порога не
    #    зависит и говорит, на каком n метрика вообще будет считаться.
    try:
        idx, f0 = strata.load_f0_estimates()
    except (FileNotFoundError, KeyError) as e:
        rep["strata"] = {"defined": False, "error": str(e)}
    else:
        keep = np.isin(idx, np.flatnonzero((yd == 1) & (spd != 0)))
        idx, f0 = idx[keep], f0[keep]
        order = np.argsort(idx)
        idx, f0 = idx[order], f0[order]
        b = strata.band_of(f0)
        n_by_band = {strata._name(j): int((b == j).sum())
                     for j in range(len(strata.F0_BANDS)) if (b == j).sum()}
        if not op["threshold_calibrated"]:
            rep["strata"] = {"defined": False, "reason": op["reason"],
                             "n_by_band": n_by_band,
                             "n_windows": int(len(idx))}
        else:
            Xp = np.ascontiguousarray(Xd[idx]).astype(np.float32) / 32768.0
            lg_pos = score_windows(scorer, Xp)
            det = strata.recall_by_band_detail(lg_pos, np.ones(len(lg_pos), int),
                                               f0, on)
            wname, wval = strata.worst_band(det)
            rep["strata"] = {"defined": True, "by_band": det,
                             "worst_band": wname, "worst_recall": wval,
                             "saturated_threshold": strata.SATURATED,
                             "n_windows": int(len(lg_pos))}

    rep["provenance"] = _provenance(scorer, name, seed, model_path, hard_cats,
                                    len(hard), len(clips), len(track), n_win,
                                    mask_raw, mask_sm)
    return rep


def _provenance(scorer, name, seed, model_path, hard_cats, n_hard_windows,
                n_clips, n_samples, n_win, mask_raw, mask_sm):
    """Всё, чем один прогон отличается от другого.

    Без этого блока замороженная базовая цифра непроверяема: через месяц
    нельзя установить, считался ли новый прогон по тому же чекпоинту, тому же
    списку трудных категорий и тому же числу окон, — а сравнение с базой
    только этим и держится.
    """
    return {
        "model_path": model_path,
        "seed": int(seed),
        "git_sha": _git_sha(),
        # SHA без этого флага врёт: при незакоммиченных правках код прогона
        # не тот, что лежит в коммите, и «сверить входы» по нему нельзя
        "git_dirty": _git_dirty(),
        "fa_budget_per_hour": FA_BUDGET,
        "off_delta": OFF_DELTA,
        "tau_s": TAU_S,
        "nominal_far": NOMINAL_FAR,
        "min_expected_events": MIN_EXPECTED_EVENTS,
        "n_rep": N_REP,
        "n_boot_snr50": N_BOOT_SNR50,
        "hard_categories": sorted(hard_cats),
        "n_hard_categories": len(hard_cats),
        "corpus": {
            "hard_holdout_windows": int(n_hard_windows),
            "clips_after_regroup": int(n_clips),
            "track_samples": int(n_samples),
            "track_windows": int(n_win),
            "windows_kept_raw": int(mask_raw.sum()),
            "windows_kept_smoothed": int(mask_sm.sum()),
            "windows_masked_out_raw": int((~mask_raw).sum()),
            "windows_masked_out_smoothed": int((~mask_sm).sum()),
        },
    }


def _git_sha():
    return _git("rev-parse", "HEAD")


def _git_dirty():
    # Спрашивается ровно одно: отличается ли КОД ХАРНЕСА от коммита, на который
    # указывает git_sha. Поэтому проверка сужена до airadar/ и cli/ и не видит
    # ни untracked-файлов, ни самого лога прогона — а лог tee успевает
    # обрезать раньше, чем сюда дойдёт очередь, и общий git status пометил бы
    # грязным каждый прогон подряд, обесценив флаг.
    st = _git("status", "--porcelain", "--untracked-files=no", "--", "airadar", "cli")
    return None if st is None else bool(st)


def _git(*args):
    try:
        out = subprocess.run(["git", *args], cwd=corpus.ROOT,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


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


def _json_safe(obj, path="", nan_paths=None):
    """NaN/inf -> null, с записью пути в отдельный список.

    Голый токен NaN не является JSON по RFC 8259, и всё, кроме Python, на нём
    падает. Просто заменить на null нельзя: пропадёт разница между «считали и
    получили неопределённость» и «поля вообще нет», а этот файл — цель для
    сравнения будущих моделей, и такую разницу он терять не имеет права.
    Поэтому null плюс перечень путей в `nan_fields`.
    """
    if nan_paths is None:
        nan_paths = []
    if isinstance(obj, dict):
        return {k: _json_safe(v, f"{path}.{k}" if path else str(k), nan_paths)
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v, f"{path}[{i}]", nan_paths)
                for i, v in enumerate(obj)]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        if not math.isfinite(v):
            nan_paths.append(path)
            return None
        return v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist(), path, nan_paths)
    return obj


def write_report(rep, out_dir="bench_out"):
    os.makedirs(out_dir, exist_ok=True)
    jp = os.path.join(out_dir, f"{rep['name']}.json")
    mp = os.path.join(out_dir, f"{rep['name']}.md")
    nan_paths = []
    safe = {k: _json_safe(v, k, nan_paths) for k, v in rep.items()}
    safe["nan_fields"] = sorted(nan_paths)
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(safe, f, ensure_ascii=False, indent=2, allow_nan=False)
    with open(mp, "w", encoding="utf-8") as f:
        f.write(_markdown(rep))
    return jp, mp


def _nan_note(rec):
    """Почему SNR50 не определён: кривая не поднимается или не опускается.

    nan сам по себе одинаково выглядит в обоих случаях, а выводы из них
    противоположные — «не обнаруживает нигде» против «обнаруживает везде».
    """
    c = rec["curve"]
    if not c:
        return "nan (пустая кривая)"
    if max(c) < 0.5:
        return f"nan (не обнаруживает ни на одной ступени, max p={max(c):.2f})"
    if min(c) >= 0.5:
        return f"nan (обнаруживает на всех ступенях, min p={min(c):.2f})"
    return f"nan (кривая не пересекает 0.5 сверху вниз: {c[0]:.2f}→{c[-1]:.2f})"


def _markdown(rep):
    op = rep["operating_point"]
    out = [f"# {rep['name']}", ""]

    if op["threshold_calibrated"]:
        out += [f"Рабочая точка **откалибрована** ({op['threshold_source']}): "
                f"бюджет {op['fa_budget_per_hour']} тревог/час, фактически "
                f"{op['fa_actual']:.2f} на {op['background_hours_scored']:.3f} ч "
                f"зачтённого фона (вся дорожка "
                f"{op['background_hours_total']:.3f} ч).",
                f"Порог включения {op['on']:.3f}, выключения {op['off']:.3f}."]
    else:
        out += ["## Рабочая точка НЕ определена", "",
                f"**{op['reason']}**", "",
                f"- зачтённого фона {op['background_hours_scored']:.3f} ч из "
                f"{op['background_hours_total']:.3f} ч дорожки "
                f"(стыки и память EMA оставили "
                f"{100*op['seam_mask_kept_smoothed']:.0f}% окон; "
                f"без учёта памяти EMA было бы "
                f"{100*op['seam_mask_kept_raw']:.0f}%);",
                f"- шаг квантования FA/час на этом фоне "
                f"{op['fa_resolution_per_hour']:.2f} тревог/час при бюджете "
                f"{op['fa_budget_per_hour']};",
                f"- ожидаемых событий при бюджете "
                f"{op['expected_events_at_budget']:.2f}, порог доверия "
                f"{op['min_expected_events']:.0f}.", "",
                "Порог **не подставлен ничем**: запасное значение "
                "`max(фон)+off_delta` тревогу не даёт никогда и как "
                "калибровка не годится. Всё, что от рабочей точки зависит "
                "(SNR50, recall по f0-полосам), ниже помечено как "
                "неопределённое."]
    out += [""]

    sn = rep["snr50"]
    if sn.get("defined"):
        out += [f"| запись | SNR50, дБ | 95% CI SNR50 | auc_fh | 95% CI auc_fh |",
                "|---|---|---|---|---|"]
        for nm, rec in sn["by_record"].items():
            f_ = rep["field"][nm]
            lo, hi = f_["auc_fh_ci"]
            s = rec["snr50_db"]
            slo, shi = rec["snr50_ci"]
            s_txt = _nan_note(rec) if not np.isfinite(s) else f"{s:.1f}"
            ci_txt = ("н/д" if not np.isfinite(slo)
                      else f"[{slo:.1f}, {shi:.1f}]")
            out.append(f"| {nm} | {s_txt} | {ci_txt} | {f_['auc_fh']:.3f} | "
                       f"[{lo:.3f}, {hi:.3f}] |")
        out += ["", f"SNR50: {sn['n_rep']} повторов на ступень, "
                    f"{sn['n_boot']} бутстрап-реплик по повторам."]
    else:
        out += ["**SNR50 не определён** — нет рабочей точки (см. выше). "
                "Ниже только то, что от неё не зависит.", "",
                "| запись | auc_fh | 95% CI auc_fh | медианный перцентиль |",
                "|---|---|---|---|"]
        for nm, f_ in rep["field"].items():
            lo, hi = f_["auc_fh_ci"]
            out.append(f"| {nm} | {f_['auc_fh']:.3f} | [{lo:.3f}, {hi:.3f}] | "
                       f"{f_['median_pct']:.3f} |")

    t = rep["transfer"]
    out += ["", f"Перенос порога DADS→трудные: номинал {t['far_nominal']:.3f}, "
                f"факт {t['far_actual']:.3f}, отношение **{t['ratio']:.1f}×**, "
                f"дрейф p99 {t['drift_p99']:.2f} σ."]

    st = rep.get("strata", {})
    if st.get("defined"):
        out += ["", "| f0-полоса, Гц | recall | 95% CI (Уилсон) | окон | "
                    "попаданий | |", "|---|---|---|---|---|---|"]
        for k, v in sorted(st["by_band"].items()):
            lo, hi = v["ci"]
            flag = ("**насыщено, не информативно**" if v["saturated"] else "")
            out.append(f"| {k} | {v['recall']:.3f} | [{lo:.3f}, {hi:.3f}] | "
                       f"{v['n']} | {v['hits']} | {flag} |")
        out += ["", f"Худшая полоса: **{st['worst_band']} Гц, "
                    f"recall {st['worst_recall']:.3f}** "
                    f"({st['n_windows']} окон всего). Это и есть отчётная "
                    f"величина — среднее по полосам скрывает именно тяжёлые "
                    f"машины.",
                "", f"Полосы с recall ≥ {st['saturated_threshold']} помечены "
                    f"как насыщенные: на таком уровне число перестаёт "
                    f"различать модели, и сравнивать по нему нельзя.",
                "", "Интервал Уилсона предполагает независимые окна. Окна "
                    "cache_dads перекрываются на 50% и приходят группами по "
                    "клипам, поэтому НАСТОЯЩИЙ интервал шире напечатанного — "
                    "это нижняя оценка разброса. Полноценный интервал требует "
                    "группировки по клипам, то есть манифеста (этап 1)."]
        out += ["", "Recall по полосам считается на сыром (несглаженном) "
                    "логите изолированного окна против того же порога `on`, "
                    "что и рабочая точка — но `on` откалиброван по "
                    "сглаженному EMA-потоку (`decision.smooth`), который "
                    "гасит одиночные всплески. Сырому мгновенному пику "
                    "проще пробить этот порог, чем сглаженному значению в "
                    "потоке, поэтому recall по полосам НЕ сравним напрямую "
                    "с `fa_actual`/`snr50` выше — те посчитаны на "
                    "сглаженном+гистерезисном пайплайне, а это число — нет."]
    elif "n_by_band" in st:
        out += ["", "**Recall по f0-полосам не определён** — он считается "
                    "против порога рабочей точки, а её нет. Состав страт от "
                    "порога не зависит и приводится как есть:", "",
                "| f0-полоса, Гц | окон |", "|---|---|"]
        out += [f"| {k} | {v} |" for k, v in sorted(st["n_by_band"].items())]
        out += ["", f"Всего {st['n_windows']} окон. На 100-150 окон в полосе "
                    f"ширина интервала Уилсона порядка ±5-7 пп, что сравнимо "
                    f"со всей разницей между полосами — это надо иметь в виду "
                    f"уже сейчас, когда полоса станет измеримой."]
    else:
        out += ["", f"Страты не посчитаны: {st.get('error', 'нет данных')}"]

    pv = rep.get("provenance", {})
    if pv:
        c = pv["corpus"]
        out += ["", "## Происхождение прогона", "",
                f"- модель: `{pv['model_path']}`",
                f"- git: `{pv['git_sha']}`"
                + (" (**рабочее дерево грязное** — код прогона не совпадает "
                   "с коммитом)" if pv.get("git_dirty") else ""),
                f"- seed: {pv['seed']}, tau={pv['tau_s']} с, "
                f"off_delta={pv['off_delta']}, бюджет "
                f"{pv['fa_budget_per_hour']} тревог/час",
                f"- корпус: {c['hard_holdout_windows']} удержанных трудных окон "
                f"из {pv['n_hard_categories']} категорий → "
                f"{c['clips_after_regroup']} клипов → дорожка "
                f"{c['track_windows']} окон;",
                f"- масок: сырая оставила {c['windows_kept_raw']} "
                f"(вырезано {c['windows_masked_out_raw']}), сглаженная — "
                f"{c['windows_kept_smoothed']} "
                f"(вырезано {c['windows_masked_out_smoothed']});",
                f"- трудные категории: {', '.join(pv['hard_categories'])}."]
    return "\n".join(out) + "\n"
