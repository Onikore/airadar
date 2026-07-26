"""Репозиторий HF как общий диск между Colab и рабочей машиной.

Colab теряет диск при обрыве сессии, а прогресс обучения надо видеть снаружи.
Всё, что должно пережить сессию — кэш окон, логи, чекпоинты — лежит здесь.

Раскладка в репозитории:
    cache/cache_dads/   окна дрона и фона + meta.npz с замороженным сплитом
    cache/cache_hard/   негативы с категориями
    field/              полевые записи (не публичные)
    runs/<имя>/         train.log, metrics.jsonl, last.pt, best.pt
    manifest.json       прогресс препроцессинга, для докачки после обрыва
"""

import os
import sys
import json

REPO = "Onikore/airadar-hub"
REPO_TYPE = "dataset"


def token():
    t = os.environ.get("HF_TOKEN")
    if not t:
        try:                                  # в Colab токен живёт в Secrets
            from google.colab import userdata
            t = userdata.get("HF_TOKEN")
        except Exception:
            t = None
    if not t:
        raise RuntimeError(
            "нет токена HF: задайте переменную окружения HF_TOKEN "
            "или добавьте секрет HF_TOKEN в Colab (значок ключа слева)")
    return t


def _norm(p):
    """Пути в HF всегда через прямой слэш и без ведущего."""
    return "/".join(x for x in str(p).replace("\\", "/").split("/") if x)


def _dump_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _api():
    from huggingface_hub import HfApi
    return HfApi(token=token())


def ensure_repo():
    """Создаёт приватный репозиторий, если его ещё нет.

    Существование проверяется ДО create_repo: у fine-grained токена без права
    создавать репозитории create_repo отдаёт 403 даже с exist_ok=True — HF
    проверяет разрешение раньше, чем существование. То есть на готовом
    репозитории вызов падал бы там, где падать нечему.
    """
    api = _api()
    if api.repo_exists(REPO, repo_type=REPO_TYPE):
        return
    api.create_repo(REPO, repo_type=REPO_TYPE, private=True, exist_ok=True)


def check_access():
    """Проверяет, что токен действительно может писать в репозиторий.

    Нужна затем, что препроцессинг идёт 40 минут, а первая выгрузка случается
    в конце первого шарда. Узнать про нехватку прав лучше на второй секунде.

    Возвращает имя пользователя. Падает с внятным сообщением, а не 403 из
    глубины httpx.
    """
    api = _api()
    try:
        me = api.whoami()
    except Exception as e:
        raise RuntimeError(f"токен HF недействителен: {e}") from e

    name = me.get("name")
    tok = me.get("auth", {}).get("accessToken", {})
    fg = tok.get("fineGrained")
    owner = REPO.split("/")[0]
    if fg is not None:
        scoped = {s["entity"]["name"]: s["permissions"] for s in fg.get("scoped", [])}
        if "repo.write" not in scoped.get(owner, []):
            raise RuntimeError(
                f"токен «{tok.get('displayName')}» (пользователь {name}) не может "
                f"писать в «{owner}»: выписан на {sorted(scoped) or 'ничего'}.\n"
                f"Нужен scope repo.write на «{owner}». Заведите токен на "
                f"huggingface.co/settings/tokens: тип Fine-grained, владелец "
                f"{owner}, галочка «Write access to contents/settings of all "
                f"repos», затем обновите секрет HF_TOKEN в Colab.")

    ensure_repo()
    probe = f"runs/_access_{name}.json"
    write_json({"проверка": "доступ на запись"}, probe)
    if read_json(probe) is None:
        raise RuntimeError("запись прошла, но чтение вернуло пусто — "
                           "проверьте права токена на чтение")
    try:
        api.delete_file(probe, REPO, repo_type=REPO_TYPE)
    except Exception:
        pass                        # не критично: файл крошечный
    return name


def exists(remote):
    try:
        files = _api().list_repo_files(REPO, repo_type=REPO_TYPE)
    except Exception:
        return False
    r = _norm(remote)
    return r in files or any(f.startswith(r + "/") for f in files)


def push(local, remote):
    api, r = _api(), _norm(remote)
    if os.path.isdir(local):
        api.upload_folder(folder_path=local, path_in_repo=r,
                          repo_id=REPO, repo_type=REPO_TYPE)
    else:
        api.upload_file(path_or_fileobj=local, path_in_repo=r,
                        repo_id=REPO, repo_type=REPO_TYPE)


def pull(remote, local):
    """Каталог отличается от файла отсутствием расширения — в этом репозитории
    все файлы его имеют (.npz, .bin, .pt, .log, .jsonl, .json, .wav)."""
    from huggingface_hub import hf_hub_download, snapshot_download
    import shutil
    r = _norm(remote)
    if os.path.splitext(r)[1] == "":
        return snapshot_download(REPO, repo_type=REPO_TYPE, token=token(),
                                 allow_patterns=f"{r}/*", local_dir=local)
    os.makedirs(os.path.dirname(os.path.abspath(local)), exist_ok=True)
    p = hf_hub_download(REPO, r, repo_type=REPO_TYPE, token=token())
    shutil.copyfile(p, local)
    return local


def read_json(remote):
    if not exists(remote):
        return None
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(REPO, _norm(remote), repo_type=REPO_TYPE, token=token())
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def write_json(obj, remote):
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "tmp.json")
        _dump_json(obj, p)
        push(p, remote)


def selfcheck():
    """В сеть не ходит — сетевой круг проверяется отдельно, вручную."""
    import tempfile

    # токен обязателен и не подставляется молча
    old = os.environ.pop("HF_TOKEN", None)
    try:
        try:
            token()
        except RuntimeError:
            pass
        else:
            raise AssertionError("без HF_TOKEN должен быть RuntimeError")
    finally:
        if old is not None:
            os.environ["HF_TOKEN"] = old

    os.environ["HF_TOKEN"] = "hf_" + "x" * 10
    assert token().startswith("hf_")
    if old is None:
        del os.environ["HF_TOKEN"]
    else:
        os.environ["HF_TOKEN"] = old

    # нормализация путей: и Windows-разделитель, и лишние слэши
    assert _norm("cache\\meta.npz") == "cache/meta.npz"
    assert _norm("/cache//meta.npz/") == "cache/meta.npz"
    assert _norm("runs/a/last.pt") == "runs/a/last.pt"

    # сериализация json детерминирована и читаема
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "a.json")
        _dump_json({"b": 1, "a": [2, 3], "ключ": "значение"}, p)
        raw = open(p, encoding="utf-8").read()
        assert json.loads(raw) == {"b": 1, "a": [2, 3], "ключ": "значение"}
        assert "\n" in raw                      # с отступами, не одной строкой
        assert "ключ" in raw                    # кириллица не в \uXXXX

    print("selfcheck ok")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else print(f"репозиторий: {REPO}")
