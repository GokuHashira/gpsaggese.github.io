"""
Loaders and normalizers that turn open math datasets into the v2 format
(`references/format_v2.md`).

Training sources (normalized to `logic_slm_format.Example`, task="math"):
- `openmathinstruct2`: NVIDIA OpenMathInstruct-2 (CC-BY-4.0), 2M subset
- `metamathqa`: MetaMathQA-395K (MIT)
- `orcamath`: Orca-Math word problems 200k (MIT)
- `gsm8k-train`: GSM8K train (MIT)

Evaluation sets (records in the same JSONL schema):
- `gsm8k-test` (1,319), `math500` (500), `svamp` (test split),
  `proofwriter` (test split, task="logic", answer yes/no/unknown)

CLI:

python logic_slm_realdata.py build-train --source metamathqa \
    --out data/processed/real/metamathqa --workers 16
python logic_slm_realdata.py build-eval --out data/processed/real/eval

Import as:

import logic_slm_realdata as lsreda
"""

import argparse
import collections
import glob
import gzip
import json
import multiprocessing as mp
import os
import re
import time
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import logic_slm_format as lsform
import logic_slm_math_m1 as lsmam1

TRAIN_SOURCES = ("openmathinstruct2", "metamathqa", "orcamath", "gsm8k-train")
EVAL_SETS = ("gsm8k-test", "math500", "svamp", "proofwriter")
DEFAULT_RAW_DIR = "data/raw"
MAX_SOLUTION_WORDS = 1500
MAX_NON_ASCII_FRAC = 0.05
MAX_ANSWER_CHARS = 100
GSM8K_OMI_SOURCES = ("gsm8k", "augmented_gsm8k")

# A normalizer returns (example, None) or (None, drop reason).
Result = Tuple[Optional[lsform.Example], Optional[str]]

_CALC_RE = re.compile(r"<<[^<>]*>>")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\\$(])")
_NUMBER_RE = re.compile(r"(?<![\w.])-?\$?\d[\d,]*(?:\.\d+)?(?:/\d+)?(?![\w])")
_THOUSANDS_RE = re.compile(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$")
_ANSWER_VERB_RE = re.compile(r"\b(?:is|are|was|were|be|equals|equal to|of)\b|=|:")
_METAMATH_ANSWER_RE = re.compile(r"The answer is:?\s*", re.IGNORECASE)


# #############################################################################
# Shared cleaning
# #############################################################################


def clean_question(q: Any) -> str:
    """
    Collapse all whitespace (including newlines) so the prompt is one line.
    """
    return " ".join(str(q or "").split())


def solution_lines(text: str) -> List[str]:
    """
    Split a solution into stripped, non-empty lines; lines that would look
    like an answer line ("####...") are dropped.
    """
    out = []
    for line in str(text or "").split("\n"):
        s = " ".join(line.split())
        if s and not s.startswith(lsform.ANSWER_MARKER):
            out.append(s)
    return out


def _boxed_content(s: str) -> Optional[str]:
    """
    Content of the last `\\boxed{...}` in `s` (balanced braces), or None.
    """
    i = s.rfind("\\boxed")
    if i < 0:
        return None
    j = s.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(s)):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                return s[j + 1 : k]
    return None


def clean_answer(a: Any) -> Optional[str]:
    """
    Light answer cleaning: whitespace, `\\boxed{}`, outer `$...$`, trailing
    ".", and for plain numbers a leading "$" and thousands separators.

    :return: cleaned answer, or None if empty / multi-line / too long
    """
    if a is None:
        return None
    s = str(a).strip()
    if not s or "\n" in s:
        return None
    s = " ".join(s.split())
    boxed = _boxed_content(s)
    if boxed is not None:
        s = boxed.strip()
    if len(s) > 2 and s.startswith("$") and s.endswith("$") and s.count("$") == 2:
        s = s[1:-1].strip()
    s = s.rstrip(".").strip()
    num = s[1:] if s.startswith("$") else s
    if _THOUSANDS_RE.match(num):
        num = num.replace(",", "")
    if re.match(r"^-?\d+(\.\d+)?$", num):
        s = num
    if not s or len(s) > MAX_ANSWER_CHARS or s.startswith(lsform.ANSWER_MARKER):
        return None
    return s


def _non_ascii_frac(text: str) -> float:
    if not text:
        return 0.0
    return sum(ord(c) > 127 for c in text) / len(text)


def make_example(
    question: str,
    lines: List[str],
    answer: Optional[str],
    meta: Dict[str, Any],
    *,
    require_steps: bool = True,
) -> Result:
    """
    Validate the pieces and build a math Example, or return a drop reason.

    Reasons: empty_question, bad_answer, empty_solution, too_long, non_ascii,
    invalid.
    """
    if not question:
        return None, "empty_question"
    if answer is None or lsform.normalize_answer(answer, "math") in (None, ""):
        return None, "bad_answer"
    if require_steps and not lines:
        return None, "empty_solution"
    if sum(len(line.split()) for line in lines) > MAX_SOLUTION_WORDS:
        return None, "too_long"
    if _non_ascii_frac(question + " ".join(lines)) > MAX_NON_ASCII_FRAC:
        return None, "non_ascii"
    try:
        ex = lsform.Example(task="math", problem=question, steps=lines, answer=answer, meta=meta)
    except ValueError:
        return None, "invalid"
    return ex, None


# #############################################################################
# Training-source normalizers
# #############################################################################


def _drop_boxed_sentence(lines: List[str]) -> List[str]:
    """
    Remove the final sentence that contains `\\boxed` (and the whole line if
    nothing else remains on it).
    """
    idx = None
    for i in range(len(lines) - 1, -1, -1):
        if "\\boxed" in lines[i]:
            idx = i
            break
    if idx is None:
        return lines
    sents = _SENT_SPLIT_RE.split(lines[idx])
    kept = [s for s in sents if "\\boxed" not in s]
    rest = " ".join(kept).strip()
    out = lines[:idx]
    if rest and re.search(r"[A-Za-z0-9]", rest):
        out.append(rest)
    return out + lines[idx + 1 :]


def normalize_openmathinstruct2(row: Dict[str, Any], *, gsm8k_only: bool = False) -> Result:
    """
    OpenMathInstruct-2 row (problem, generated_solution, expected_answer,
    problem_source) -> Example. The final `\\boxed{}` sentence is removed and
    the answer is `expected_answer`.

    :param gsm8k_only: keep only gsm8k / augmented_gsm8k problems
    """
    src = str(row.get("problem_source") or "")
    if gsm8k_only and src not in GSM8K_OMI_SOURCES:
        return None, "not_gsm8k_source"
    lines = _drop_boxed_sentence(solution_lines(row.get("generated_solution")))
    meta = {"level": "openmathinstruct2", "problem_source": src}
    return make_example(
        clean_question(row.get("problem")), lines, clean_answer(row.get("expected_answer")), meta
    )


def unbox_all(s: str) -> str:
    """
    Replace every `\\boxed{X}` in `s` by `X` (balanced braces), keeping the
    sentence: MetaMathQA solutions box values inside their explanations.
    """
    out, i = [], 0
    while True:
        j = s.find("\\boxed{", i)
        if j < 0:
            out.append(s[i:])
            return "".join(out)
        out.append(s[i:j])
        k, depth = j + len("\\boxed{"), 1
        while k < len(s) and depth:
            depth += {"{": 1, "}": -1}.get(s[k], 0)
            k += 1
        if depth:  # unbalanced: keep the rest unchanged
            out.append(s[j:])
            return "".join(out)
        out.append(s[j + len("\\boxed{") : k - 1])
        i = k


def normalize_metamathqa(row: Dict[str, Any]) -> Result:
    """
    MetaMathQA row (query, response, type) -> Example; the answer follows
    the last "The answer is:".
    """
    resp = str(row.get("response") or "")
    matches = list(_METAMATH_ANSWER_RE.finditer(resp))
    if not matches:
        return None, "bad_answer"
    m = matches[-1]
    answer = clean_answer(resp[m.end() :].strip())
    lines = [unbox_all(line) for line in solution_lines(resp[: m.start()])]
    meta = {"level": "metamathqa", "type": str(row.get("type") or "")}
    return make_example(clean_question(row.get("query")), lines, answer, meta)


def extract_final_number(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract a clean final numeric answer from a free-text solution.

    Rules (conservative): use the last `\\boxed{}` if any; otherwise look at
    the last sentence of the last line: if it has exactly one number, that is
    the answer; otherwise, if exactly one number follows the last
    "is / are / was / = / :" in the sentence, that is the answer.

    :return: (answer, None) or (None, reason) with reason no_final_answer or
        ambiguous_final_answer
    """
    lines = solution_lines(text.replace("**", ""))
    if not lines:
        return None, "no_final_answer"
    boxed = _boxed_content(lines[-1])
    if boxed is not None:
        a = clean_answer(boxed)
        if a is not None and _NUMBER_RE.fullmatch(a.replace("$", "")):
            return a, None
        return None, "ambiguous_final_answer"
    sent = _SENT_SPLIT_RE.split(lines[-1])[-1]
    nums = _NUMBER_RE.findall(sent)
    if not nums:
        return None, "no_final_answer"
    if len(nums) > 1:
        verbs = list(_ANSWER_VERB_RE.finditer(sent))
        tail_nums = _NUMBER_RE.findall(sent[verbs[-1].end() :]) if verbs else []
        if len(tail_nums) != 1:
            return None, "ambiguous_final_answer"
        nums = tail_nums
    a = clean_answer(nums[0].replace("$", ""))
    if a is None:
        return None, "no_final_answer"
    return a, None


def normalize_orcamath(row: Dict[str, Any]) -> Result:
    """
    Orca-Math row (question, answer) -> Example; the final numeric answer is
    extracted from the answer text (dropped if there is no clean one).
    """
    text = str(row.get("answer") or "")
    answer, reason = extract_final_number(text)
    if answer is None:
        return None, reason
    lines = solution_lines(text.replace("**", ""))
    meta = {"level": "orcamath"}
    return make_example(clean_question(row.get("question")), lines, answer, meta)


def split_gsm8k_answer(text: str) -> Tuple[List[str], Optional[str]]:
    """
    Split a GSM8K answer into solution lines (calculator annotations
    `<<...>>` removed) and the final answer after "####".
    """
    text = str(text or "")
    pos = text.rfind(lsform.ANSWER_MARKER)
    if pos < 0:
        return solution_lines(_CALC_RE.sub("", text)), None
    answer = clean_answer(text[pos + len(lsform.ANSWER_MARKER) :].strip())
    return solution_lines(_CALC_RE.sub("", text[:pos])), answer


def normalize_gsm8k(row: Dict[str, Any], *, level: str = "gsm8k-train") -> Result:
    """
    GSM8K row (question, answer) -> Example.
    """
    lines, answer = split_gsm8k_answer(row.get("answer"))
    return make_example(clean_question(row.get("question")), lines, answer, {"level": level})


def example_to_record(ex: lsform.Example, gen: str) -> Dict[str, Any]:
    """
    JSONL record: the fields of `logic_slm_math_m1.example_to_record` plus
    `gen` (source name).
    """
    rec = lsmam1.example_to_record(ex)
    rec["gen"] = gen
    return rec


# #############################################################################
# Raw readers
# #############################################################################


def _parquet_files(pattern: str) -> List[str]:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files match {pattern}")
    return files


def iter_parquet(path: str, columns: Optional[List[str]] = None, row_group: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    """
    Iterate the rows of a parquet file (or one row group) as dicts.
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    groups = [row_group] if row_group is not None else range(pf.num_row_groups)
    for g in groups:
        tbl = pf.read_row_group(g, columns=columns)
        yield from tbl.to_pylist()


def _source_spec(source: str, raw_dir: str) -> Tuple[str, List[str]]:
    """
    (glob pattern, columns) of a parquet training source.
    """
    if source == "openmathinstruct2":
        return (
            os.path.join(raw_dir, "openmathinstruct2/data/train_2M-*.parquet"),
            ["problem", "generated_solution", "expected_answer", "problem_source"],
        )
    if source == "orcamath":
        return os.path.join(raw_dir, "orcamath/data/train-*.parquet"), ["question", "answer"]
    if source == "gsm8k-train":
        return os.path.join(raw_dir, "gsm8k/main/train-*.parquet"), ["question", "answer"]
    raise ValueError(f"Not a parquet source: {source}")


def get_normalizer(source: str, *, gsm8k_only: bool = False) -> Callable[[Dict[str, Any]], Result]:
    """
    Row normalizer of a training source.
    """
    if source == "openmathinstruct2":
        return lambda r: normalize_openmathinstruct2(r, gsm8k_only=gsm8k_only)
    if source == "metamathqa":
        return normalize_metamathqa
    if source == "orcamath":
        return normalize_orcamath
    if source == "gsm8k-train":
        return normalize_gsm8k
    raise ValueError(f"Unknown source '{source}'; expected one of {TRAIN_SOURCES}")


def normalize_rows(
    rows: Iterable[Dict[str, Any]], source: str, *, gsm8k_only: bool = False
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Normalize raw rows into records.

    :return: (records, stats {"in", "kept", "drop:<reason>"})
    """
    fn = get_normalizer(source, gsm8k_only=gsm8k_only)
    stats: Dict[str, int] = collections.Counter()
    recs = []
    for row in rows:
        stats["in"] += 1
        ex, reason = fn(row)
        if ex is None:
            stats[f"drop:{reason}"] += 1
            continue
        stats["kept"] += 1
        recs.append(example_to_record(ex, source))
    return recs, dict(stats)


# #############################################################################
# build-train
# #############################################################################


def _write_jsonl_gz(path: str, recs: Iterable[Dict[str, Any]]) -> None:
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt", compresslevel=6) as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _train_job(job: Dict[str, Any]) -> Dict[str, int]:
    """
    Worker: normalize one chunk (parquet row group or list of rows) and
    write it as one shard.
    """
    if "rows" in job:
        rows: Iterable[Dict[str, Any]] = job["rows"]
    else:
        rows = iter_parquet(job["path"], job["columns"], job["row_group"])
    recs, stats = normalize_rows(rows, job["source"], gsm8k_only=job["gsm8k_only"])
    _write_jsonl_gz(job["out"], recs)
    return stats


def _train_jobs(source: str, raw_dir: str, out_dir: str, gsm8k_only: bool, chunk: int) -> List[Dict[str, Any]]:
    """
    Split a source into shard jobs.
    """
    jobs: List[Dict[str, Any]] = []
    base = {"source": source, "gsm8k_only": gsm8k_only}
    if source == "metamathqa":
        path = os.path.join(raw_dir, "metamathqa/MetaMathQA-395K.json")
        with open(path) as f:
            rows = json.load(f)
        for i in range(0, len(rows), chunk):
            jobs.append({**base, "rows": rows[i : i + chunk]})
    else:
        import pyarrow.parquet as pq

        pattern, columns = _source_spec(source, raw_dir)
        for path in _parquet_files(pattern):
            for g in range(pq.ParquetFile(path).num_row_groups):
                jobs.append({**base, "path": path, "columns": columns, "row_group": g})
    for i, job in enumerate(jobs):
        job["out"] = os.path.join(out_dir, f"part-{i:05d}.jsonl.gz")
    return jobs


def build_train(
    source: str,
    out_dir: str,
    *,
    raw_dir: str = DEFAULT_RAW_DIR,
    workers: int = 1,
    gsm8k_only: bool = False,
    chunk: int = 50_000,
) -> Dict[str, int]:
    """
    Normalize a training source into gzip JSONL shards in `out_dir`.

    :return: aggregated stats (also written to `out_dir/stats.json`)
    """
    get_normalizer(source)
    os.makedirs(out_dir, exist_ok=True)
    for old in glob.glob(os.path.join(out_dir, "part-*.jsonl.gz")):
        os.remove(old)
    jobs = _train_jobs(source, raw_dir, out_dir, gsm8k_only, chunk)
    total: Dict[str, int] = collections.Counter()
    if workers <= 1 or len(jobs) == 1:
        for job in jobs:
            total.update(_train_job(job))
    else:
        with mp.get_context("fork").Pool(min(workers, len(jobs))) as pool:
            for st in pool.imap_unordered(_train_job, jobs):
                total.update(st)
    total = dict(sorted(total.items()))
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump({"source": source, "gsm8k_only": gsm8k_only, "shards": len(jobs), **total}, f, indent=2)
    return total


# #############################################################################
# Evaluation sets
# #############################################################################


def _number_str(x: Any) -> str:
    """
    Render a numeric gold answer without a spurious ".0".
    """
    f = float(x)
    if f.is_integer():
        return str(int(f))
    return format(f, ".10g")


def _eval_record(
    task: str,
    gen: str,
    prompt: str,
    answer: str,
    solution: str = "",
    **extra: Any,
) -> Dict[str, Any]:
    return {
        "format": lsform.FORMAT_VERSION,
        "task": task,
        "prompt": prompt,
        "solution": solution,
        "answer": answer,
        "level": gen,
        "gen": gen,
        **extra,
    }


def load_gsm8k_test(raw_dir: str = DEFAULT_RAW_DIR) -> List[Dict[str, Any]]:
    """
    GSM8K test (1,319 problems) with gold solutions in v2 format.
    """
    out = []
    for path in _parquet_files(os.path.join(raw_dir, "gsm8k/main/test-*.parquet")):
        for i, row in enumerate(iter_parquet(path, ["question", "answer"])):
            lines, answer = split_gsm8k_answer(row["answer"])
            ex = lsform.Example(task="math", problem=clean_question(row["question"]), steps=lines, answer=answer)
            out.append(
                _eval_record(
                    "math",
                    "gsm8k-test",
                    lsform.render_prompt(ex),
                    answer,
                    lsform.render_solution(ex),
                    id=f"gsm8k-test-{len(out)}",
                )
            )
    return out


def load_math500(raw_dir: str = DEFAULT_RAW_DIR) -> List[Dict[str, Any]]:
    """
    MATH-500 (500 problems); answers are kept as LaTeX strings.
    """
    out = []
    with open(os.path.join(raw_dir, "math500/test.jsonl")) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            ex = lsform.Example(task="math", problem=clean_question(row["problem"]), steps=[], answer=str(row["answer"]).strip())
            out.append(
                _eval_record(
                    "math",
                    "math500",
                    lsform.render_prompt(ex),
                    ex.answer,
                    id=row.get("unique_id", f"math500-{len(out)}"),
                    subject=row.get("subject"),
                    difficulty=row.get("level"),
                )
            )
    return out


def load_svamp(raw_dir: str = DEFAULT_RAW_DIR, split: str = "test") -> List[Dict[str, Any]]:
    """
    SVAMP split; question = Body + " " + Question.
    """
    out = []
    for path in _parquet_files(os.path.join(raw_dir, f"svamp/data/{split}-*.parquet")):
        for row in iter_parquet(path):
            q = clean_question(f"{str(row['Body']).strip()} {str(row['Question']).strip()}")
            ex = lsform.Example(task="math", problem=q, steps=[], answer=_number_str(row["Answer"]))
            out.append(
                _eval_record(
                    "math",
                    "svamp",
                    lsform.render_prompt(ex),
                    ex.answer,
                    id=str(row.get("ID", f"svamp-{len(out)}")),
                    equation=str(row.get("Equation", "")),
                    type=str(row.get("Type", "")),
                )
            )
    return out


PROOFWRITER_LABELS = {"true": "yes", "false": "no", "unknown": "unknown"}


def proofwriter_record(row: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
    """
    ProofWriter row -> logic record, or None if its label is not
    True / False / Unknown.

    SCHEMA NOT VERIFIED: written without access to the downloaded files.
    The assumed `tasksource/proofwriter` columns are `theory`, `question`,
    `answer` ("True" / "False" / "Unknown"), optional `id`, `QDep`, `maxD`,
    `config`. Check the real parquet columns and label values before use;
    rows with other labels are dropped silently.

    Premises are the natural-language theory; the question is the statement
    without its final "." (rendered as "Question: <statement>?").
    """
    label = PROOFWRITER_LABELS.get(str(row.get("answer", "")).strip().lower())
    if label is None:
        return None
    theory = clean_question(row.get("theory"))
    question = clean_question(row.get("question")).rstrip(".").rstrip("?").strip()
    if not theory or not question:
        return None
    ex = lsform.Example(task="logic", problem=theory, question=question, steps=[], answer=label)
    extra = {"id": str(row.get("id", f"proofwriter-{idx}"))}
    for key in ("QDep", "maxD", "config"):
        if key in row and row[key] is not None:
            extra[key.lower()] = row[key]
    return _eval_record("logic", "proofwriter", lsform.render_prompt(ex), label, **extra)


def load_proofwriter(raw_dir: str = DEFAULT_RAW_DIR, split: str = "test") -> List[Dict[str, Any]]:
    """
    ProofWriter split as task="logic" records with yes/no/unknown answers.
    """
    out = []
    for path in _parquet_files(os.path.join(raw_dir, f"proofwriter/data/{split}-*.parquet")):
        for row in iter_parquet(path):
            rec = proofwriter_record(row, len(out))
            if rec is not None:
                out.append(rec)
    return out


EVAL_LOADERS: Dict[str, Callable[[str], List[Dict[str, Any]]]] = {
    "gsm8k-test": load_gsm8k_test,
    "math500": load_math500,
    "svamp": load_svamp,
    "proofwriter": load_proofwriter,
}


def build_eval(out_dir: str, *, raw_dir: str = DEFAULT_RAW_DIR, sets: Iterable[str] = EVAL_SETS) -> Dict[str, int]:
    """
    Write one `<set>.jsonl` per evaluation set into `out_dir`.

    :return: record count per set
    """
    os.makedirs(out_dir, exist_ok=True)
    counts = {}
    for name in sets:
        recs = EVAL_LOADERS[name](raw_dir)
        with open(os.path.join(out_dir, f"{name}.jsonl"), "w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts[name] = len(recs)
    return counts


# #############################################################################
# CLI
# #############################################################################


def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("build-train", help="normalize a training source")
    tr.add_argument("--source", required=True, choices=TRAIN_SOURCES)
    tr.add_argument("--out", required=True)
    tr.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    tr.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    tr.add_argument("--gsm8k-only", action="store_true", help="openmathinstruct2: gsm8k-derived problems only")
    tr.add_argument("--chunk", type=int, default=50_000, help="rows per shard (metamathqa)")
    ev = sub.add_parser("build-eval", help="write the evaluation sets")
    ev.add_argument("--out", default="data/processed/real/eval")
    ev.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    ev.add_argument("--sets", nargs="+", default=list(EVAL_SETS), choices=EVAL_SETS)
    a = ap.parse_args()
    t0 = time.time()
    if a.cmd == "build-train":
        st = build_train(
            a.source,
            a.out,
            raw_dir=a.raw_dir,
            workers=a.workers,
            gsm8k_only=a.gsm8k_only,
            chunk=a.chunk,
        )
        n_in = st.get("in", 0)
        print(f"[{a.source}] in={n_in:,} kept={st.get('kept', 0):,} ({time.time() - t0:.0f}s)")
        for k, v in st.items():
            if k.startswith("drop:"):
                print(f"  {k:32s} {v:>10,} ({100 * v / max(n_in, 1):.2f}%)")
    else:
        counts = build_eval(a.out, raw_dir=a.raw_dir, sets=a.sets)
        print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    _main()
