"""
ProofWriter (OWA) data for SFT and evaluation.

- SFT pairs: rows of the `train` split (configs `depth-0` .. `depth-3` by
  default) turned into worked solutions from the dataset's own proofs
  (`allProofs`). Every derived fact becomes one step in the style of the
  synthetic L3 solutions, `<premise>, <premise>, <rule> => <conclusion>`:
  - `yes`: the proof of the question;
  - `no`: the proof of the question's negation;
  - `unknown`: every derived fact (the forward-chaining closure), as in the
    synthetic `unknown` solutions.
- Eval set: the `test` split of config `depth-5`, stratified by question
  depth (`QDep` 0..5) so accuracy can be reported per proof depth (depths 4
  and 5 are deeper than any SFT proof).

`allProofs` lists facts by depth, `@0: <fact>.[(<proof>)] ... @1: ...`. A
proof is an OR of alternatives; `tripleK` is the K-th fact of the theory and
`((<premises>) -> ruleK)` applies the K-th rule (rules numbered in theory
order) to the conclusions of `<premises>`.

Import as:

import logic_slm_proofwriter as lspw
"""

import glob
import os
import random
import re
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

import logic_slm_format as lsform
import logic_slm_mixture as lsmix
import logic_slm_realdata as lsreal

SFT_CONFIGS = ("depth-0", "depth-1", "depth-2", "depth-3")
EVAL_CONFIG = "depth-5"
EVAL_DEPTHS = (0, 1, 2, 3, 4, 5)

# A parsed proof node: ("T", k) fact k; ("R", premises, k) rule k applied to
# premise nodes.
Node = tuple

_TOKEN = re.compile(r"\(|\)|->|OR|triple\d+|rule\d+")
_ENTRY = re.compile(r"\s*([^\[\]@]+?)\s*\[([^\[\]]*)\]")


# #############################################################################
# Proof parsing
# #############################################################################


class _Parser:
    """
    Recursive-descent parser for one `allProofs` proof, e.g.
    `(triple2 OR ((triple3) -> rule3))`.
    """

    def __init__(self, text: str) -> None:
        self.toks = _TOKEN.findall(text)
        self.i = 0

    def _peek(self) -> Optional[str]:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def _next(self) -> str:
        tok = self._peek()
        if tok is None:
            raise ValueError("unexpected end of proof")
        self.i += 1
        return tok

    def group(self) -> List[List[Node]]:
        """
        `( alt OR alt ... )` -> list of alternatives (each a list of premises).
        """
        if self._next() != "(":
            raise ValueError("expected '('")
        alts = [self._seq()]
        while self._peek() == "OR":
            self._next()
            alts.append(self._seq())
        if self._next() != ")":
            raise ValueError("expected ')'")
        return alts

    def _seq(self) -> List[Node]:
        items = []
        while self._peek() not in (")", "OR", None):
            items.extend(self._item())
        return items

    def _item(self) -> List[Node]:
        tok = self._peek()
        if tok.startswith("triple"):
            self._next()
            return [("T", int(tok[6:]))]
        if tok == "(":
            alts = self.group()
            if self._peek() == "->":
                self._next()
                rule = self._next()
                if not rule.startswith("rule"):
                    raise ValueError("expected a rule after '->'")
                return [("R", tuple(alts[0]), int(rule[4:]))]
            # A bracketed group without a rule: keep its first alternative.
            return list(alts[0])
        raise ValueError(f"unexpected token {tok!r}")


def parse_proof(text: str) -> List[Node]:
    """
    Parse one proof into its alternatives (each a single node).
    """
    p = _Parser(text)
    alts = p.group()
    if p._peek() is not None:
        raise ValueError("trailing tokens in proof")
    out = []
    for alt in alts:
        if len(alt) != 1:
            raise ValueError("an alternative must be one fact or one rule application")
        out.append(alt[0])
    return out


def parse_all_proofs(text: str) -> List[Tuple[int, str, List[Node]]]:
    """
    `allProofs` -> [(depth, fact sentence, proof alternatives)].
    """
    out = []
    pieces = re.split(r"@(\d+):", text)
    for depth, body in zip(pieces[1::2], pieces[2::2]):
        for sent, proof in _ENTRY.findall(body):
            out.append((int(depth), sent.strip(), parse_proof(proof)))
    return out


# #############################################################################
# Theory structure and solutions
# #############################################################################


def split_sentences(theory: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=\.)\s+", theory.strip()) if s.strip()]


def _strip(s: str) -> str:
    return s.strip().rstrip(".")


def _canon(s: str) -> Tuple[Tuple[str, ...], bool]:
    """
    Polarity-free canonical form of a fact: ("the dog does not need the bear")
    and ("the dog needs the bear") share tokens; the bool is True if negated.
    """
    words = _strip(s).lower().split()
    neg = "not" in words
    toks = []
    for w in words:
        if w in ("not", "does", "do"):
            continue
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        toks.append(w)
    return tuple(toks), neg


def is_negation(a: str, b: str) -> bool:
    ca, na = _canon(a)
    cb, nb = _canon(b)
    return ca == cb and na != nb


class Theory:
    """
    Facts, rules and the derived closure of one ProofWriter row.

    :raises ValueError: if the proofs cannot be matched to the theory
    """

    def __init__(self, row: Dict[str, Any]) -> None:
        self.closure = parse_all_proofs(row["allProofs"])
        self.facts: Dict[int, str] = {}
        for depth, sent, alts in self.closure:
            for node in alts:
                if node[0] == "T":
                    self.facts[node[1]] = sent
        fact_sents = {_strip(s) for s in self.facts.values()}
        self.rules = [s for s in split_sentences(row["theory"]) if _strip(s) not in fact_sents]
        n_rule = row.get("NRule")
        if n_rule is not None and len(self.rules) != n_rule:
            raise ValueError(f"found {len(self.rules)} rules, expected {n_rule}")
        # Every proof alternative -> the fact it proves (used to name the
        # conclusion of nested rule applications).
        self.concl: Dict[Node, str] = {}
        for _, sent, alts in self.closure:
            for node in alts:
                self.concl.setdefault(node, sent)

    def _sentence(self, node: Node) -> str:
        if node[0] == "T":
            return self.facts[node[1]]
        return self.concl[node]

    def _emit(self, node: Node, lines: List[str], done: Set[Node]) -> None:
        """
        Append the steps proving `node` (premises first), once each.
        """
        if node[0] == "T" or node in done:
            return
        _, premises, k = node
        for p in premises:
            self._emit(p, lines, done)
        if not 1 <= k <= len(self.rules):
            raise ValueError(f"rule{k} out of range")
        parts = [_strip(self._sentence(p)) for p in premises] + [_strip(self.rules[k - 1])]
        lines.append(f"{', '.join(parts)} => {_strip(self.concl[node])}")
        done.add(node)

    def proof_lines(self, sentence: str) -> Optional[List[str]]:
        """
        Steps proving `sentence` (None if it is not in the closure). A given
        fact is restated as its own single step.
        """
        for _, sent, alts in self.closure:
            if _strip(sent) == _strip(sentence):
                node = alts[0]
                if node[0] == "T":
                    return [_strip(sent)]
                lines: List[str] = []
                self._emit(node, lines, set())
                return lines
        return None

    def closure_lines(self) -> List[str]:
        """
        Steps deriving every fact of depth >= 1.
        """
        lines: List[str] = []
        done: Set[Node] = set()
        for depth, _, alts in self.closure:
            if depth >= 1:
                self._emit(alts[0], lines, done)
        return lines


def solution_steps(row: Dict[str, Any]) -> List[str]:
    """
    Worked solution steps of one row (see the module docstring).

    :raises ValueError: if no consistent proof can be built
    """
    th = Theory(row)
    label = lsreal.PROOFWRITER_LABELS[row["answer"].strip().lower()]
    question = row["question"]
    if label == "yes":
        lines = th.proof_lines(question)
    elif label == "no":
        negs = [s for _, s, _ in th.closure if is_negation(s, question)]
        lines = th.proof_lines(negs[0]) if negs else None
    else:
        lines = th.closure_lines()
        if any(_strip(s) == _strip(question) or is_negation(s, question) for _, s, _ in th.closure):
            raise ValueError("unknown question is decided by the closure")
    if lines is None:
        raise ValueError("no proof of the answer in allProofs")
    return lines


def to_example(row: Dict[str, Any]) -> lsform.Example:
    label = lsreal.PROOFWRITER_LABELS[row["answer"].strip().lower()]
    question = lsreal.clean_question(row["question"]).rstrip(".").rstrip("?").strip()
    return lsform.Example(task="logic", problem=lsreal.clean_question(row["theory"]),
                          question=question, steps=solution_steps(row), answer=label,
                          meta={"qdep": row.get("QDep"), "config": row.get("config")})


# #############################################################################
# Data loading
# #############################################################################


def _rows(raw_dir: str, split: str, configs: Sequence[str]) -> Iterator[Dict[str, Any]]:
    pattern = os.path.join(raw_dir, f"proofwriter/data/{split}-*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files match {pattern}")
    want = set(configs)
    for path in files:
        for row in lsreal.iter_parquet(path):
            if row.get("config") in want:
                yield row


def load_sft_pairs(
    raw_dir: str, n: int, n_val: int, seed: int, exclude_theories: Set[str] = frozenset(),
    configs: Sequence[str] = SFT_CONFIGS,
) -> Tuple[List[Tuple[str, str]], List[dict], Dict[str, int]]:
    """
    Sample SFT pairs (and validation records) from the ProofWriter train split.

    Rows whose theory is in `exclude_theories` (the eval theories) are skipped.

    :return: (training pairs, validation records, {"proofwriter": n, "proofwriter_skipped": k})
    """
    rows = [r for r in _rows(raw_dir, "train", configs)
            if lsreal.clean_question(r["theory"]) not in exclude_theories]
    random.Random(lsmix.stable_seed(seed, "sft-proofwriter")).shuffle(rows)
    pairs: List[Tuple[str, str]] = []
    val: List[dict] = []
    skipped = 0
    for row in rows:
        if len(pairs) >= n and len(val) >= n_val:
            break
        try:
            ex = to_example(row)
        except (ValueError, KeyError, IndexError):
            skipped += 1
            continue
        prompt = lsform.render_prompt(ex)
        if len(val) < n_val:
            val.append({"task": "logic", "gen": "proofwriter", "prompt": prompt,
                        "answer": ex.answer, "solution": lsform.render_solution(ex)})
        else:
            pairs.append((prompt, lsform.render_solution(ex)))
    return pairs, val, {"proofwriter": len(pairs), "proofwriter_skipped": skipped}


def build_eval(raw_dir: str, per_depth: int, seed: int,
               config: str = EVAL_CONFIG, depths: Sequence[int] = EVAL_DEPTHS) -> List[dict]:
    """
    Test-split eval records stratified by question depth (`per_depth` each).
    """
    by: Dict[int, List[Dict[str, Any]]] = {d: [] for d in depths}
    for row in _rows(raw_dir, "test", [config]):
        if row.get("QDep") in by:
            by[row["QDep"]].append(row)
    out = []
    for d in depths:
        rows = by[d]
        random.Random(lsmix.stable_seed(seed, "pw-eval", d)).shuffle(rows)
        for row in rows[:per_depth]:
            rec = lsreal.proofwriter_record(row, len(out))
            if rec is not None:
                out.append(rec)
    return out


def eval_theories(records: Sequence[dict]) -> Set[str]:
    """
    Theory texts of eval records (prompt line "Premises: <theory>").
    """
    out = set()
    for r in records:
        first = r["prompt"].split("\n", 1)[0]
        if first.startswith("Premises: "):
            out.add(first[len("Premises: "):])
    return out
