"""
Logic generators with forward-chaining solutions (see
`references/format_v1.md`, section 3).

- L1 propositional: atoms `p q r s t u`, connectives `~ & | ->`.
  Premise forms: literal facts (`p`, `~q`), rules `l -> l`, `l & l -> l`,
  conjunction facts `l & l` and disjunction facts `l | l`.
- L2 first-order fragment: lowercase constants (`tom`), capitalized unary
  predicates, rules `forall x (A(x) -> B(x))`, `forall x (A(x) -> ~B(x))`,
  `forall x (A(x) & B(x) -> C(x))` (head may be negated) and ground facts
  `A(tom)`, `~B(ann)`.
- L3 English: L1 / L2 problems rendered with English templates (several per
  construct; a `template_split` holds whole templates out for a transfer
  test). The verdict and the steps come from the symbolic source.

The question is always a positive atom (`r`, `Mammal(tom)`). Steps use five
inference rules: modus ponens, modus tollens, and-elimination,
and-introduction (only to build a conjunctive rule body) and disjunctive
syllogism. `yes` / `no` solutions are the forward-chaining proof of the
question / its negation; `unknown` solutions list the whole forward-chaining
closure. Every verdict is confirmed with z3 (premises consistent, `yes` iff
premises |= q, `no` iff premises |= ~q) and examples where forward chaining
and z3 disagree are rejected, so the steps always justify the label.

Import as:

import logic_slm_logic as lslog
"""

import argparse
import dataclasses
import functools
import json
import random
import re
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple

import z3

import logic_slm_format as lsform

# A formula is a nested tuple:
#   ("v", name)              propositional atom
#   ("P", pred, term)        unary predicate applied to a constant / variable
#   ("~", f), ("&", f, g), ("|", f, g), ("->", f, g)
#   ("forall", var, f), ("exists", var, f)
Formula = tuple
# A step is (antecedent formulas, conclusion, inference rule name).
Step = Tuple[Tuple[Formula, ...], Formula, str]

LEVELS = ("L1", "L2", "L3")
VERDICTS = lsform.LOGIC_VERDICTS
TEMPLATE_SPLITS = ("train", "heldout")
INFERENCE_RULES = ("mp", "mt", "and_elim", "and_intro", "ds")
ATOMS = ("p", "q", "r", "s", "t", "u")
# Only used when a deep chain needs more than 6 atoms.
EXTRA_ATOMS = ("v", "w", "y", "z", "a", "b", "c", "d", "e", "f", "g", "h", "k", "m", "n")
CONSTANTS = (
    "tom", "ann", "bob", "sue", "max", "eve", "dan", "amy",
    "joe", "kim", "liz", "ray", "ned", "pam", "sam", "ivy",
)  # fmt: skip
VAR = "x"

# #############################################################################
# Lexicon (L2 predicates and L3 English)
# #############################################################################

# Nouns with regular "+s" plurals (or an explicit plural).
_NOUNS = (
    "cat", "dog", "mammal", "animal", "bird", "reptile", "student", "teacher",
    "doctor", "artist", "singer", "robot", "farmer", "pilot", "poet",
    "painter", "dancer", "lawyer", "nurse", "chef", "athlete", "wizard",
    "knight", "plant", "tree", "flower", "insect", "owl", "frog", "horse",
    "musician", "writer", "player", "baker", "sailor", "monkey", "rabbit",
    "engineer", "giant", "hero",
)  # fmt: skip
_PLURALS = {"hero": "heroes"}
_ADJECTIVES = (
    "happy", "kind", "red", "big", "small", "fast", "smart", "young", "old",
    "green", "cold", "quiet", "brave", "rich", "tall", "calm", "shy", "loud",
    "wise", "strong", "round", "blue", "funny", "lazy", "busy", "polite",
    "clever", "gentle", "honest", "proud",
)  # fmt: skip


@dataclasses.dataclass(frozen=True)
class _Word:
    """
    English forms of an L2 predicate.

    :param is_sg: complement after "is" ("a cat", "happy")
    :param is_pl: complement after "are" ("cats", "happy")
    :param sg: singular subject noun phrase ("cat", "happy thing")
    :param pl: plural subject noun phrase ("cats", "happy things")
    """

    is_sg: str
    is_pl: str
    sg: str
    pl: str


def _build_lexicon() -> Dict[str, _Word]:
    lex = {}
    for n in _NOUNS:
        art = "an" if n[0] in "aeiou" else "a"
        pl = _PLURALS.get(n, n + "s")
        lex[n.capitalize()] = _Word(f"{art} {n}", pl, n, pl)
    for a in _ADJECTIVES:
        lex[a.capitalize()] = _Word(a, a, f"{a} thing", f"{a} things")
    return lex


LEXICON = _build_lexicon()
PREDICATES = tuple(LEXICON)

# (positive, negative) English clauses for L1 atoms.
PROPOSITIONS = (
    ("it rains", "it does not rain"),
    ("the ground is wet", "the ground is not wet"),
    ("the sun shines", "the sun does not shine"),
    ("the store is open", "the store is not open"),
    ("the alarm rings", "the alarm does not ring"),
    ("the road is icy", "the road is not icy"),
    ("the game is cancelled", "the game is not cancelled"),
    ("the lights are on", "the lights are not on"),
    ("the door is locked", "the door is not locked"),
    ("the cat is hungry", "the cat is not hungry"),
    ("the train is late", "the train is not late"),
    ("the meeting starts", "the meeting does not start"),
    ("the battery is low", "the battery is not low"),
    ("the printer works", "the printer does not work"),
    ("the river floods", "the river does not flood"),
    ("the class is full", "the class is not full"),
    ("the coffee is hot", "the coffee is not hot"),
    ("the window is open", "the window is not open"),
    ("the phone is charged", "the phone is not charged"),
    ("the bus arrives", "the bus does not arrive"),
    ("the team wins", "the team does not win"),
    ("the oven is on", "the oven is not on"),
    ("the plant grows", "the plant does not grow"),
    ("the bridge is closed", "the bridge is not closed"),
    ("the music is loud", "the music is not loud"),
    ("the field is dry", "the field is not dry"),
    ("the dog barks", "the dog does not bark"),
    ("the market is busy", "the market is not busy"),
    ("the server is down", "the server is not down"),
    ("the test is easy", "the test is not easy"),
)

# Templates per construct: {construct: {split: [(template_id, format), ...]}}.
# Placeholders: L1 {a} {b} {c} are literal clauses; L2 {X_is} {X_are}
# {X_sg} {X_pl} are predicate forms; {l} is a literal clause; {c} a name.
_TEMPLATE_TEXT = {
    "fact": (
        ["{l}"],
        ["it is known that {l}"],
    ),
    "l1.rule": (
        ["if {a}, then {b}", "{b} if {a}", "whenever {a}, {b}", "{a} implies that {b}"],
        ["{b} provided that {a}", "in case {a}, {b}"],
    ),
    "l1.rule2": (
        ["if {a} and {b}, then {c}", "{c} if both {a} and {b}", "whenever {a} and {b}, {c}"],
        ["{c} provided that {a} and {b}"],
    ),
    "l1.and": (
        ["{a} and {b}", "both {a} and {b}"],
        ["{a}, and also {b}"],
    ),
    "l1.or": (
        ["{a} or {b}", "either {a} or {b}"],
        ["{a} or {b}, or both"],
    ),
    "l1.question": (
        ["is it true that {a}", "does it follow that {a}"],
        ["can we conclude that {a}"],
    ),
    "l2.rule": (
        [
            "all {A_pl} are {B_are}",
            "every {A_sg} is {B_is}",
            "if something is {A_is}, then it is {B_is}",
        ],
        ["anything that is {A_is} is {B_is}", "{A_pl} are always {B_are}"],
    ),
    "l2.nrule": (
        [
            "no {A_pl} are {B_are}",
            "no {A_sg} is {B_is}",
            "if something is {A_is}, then it is not {B_is}",
        ],
        ["anything that is {A_is} is not {B_is}", "{A_pl} are never {B_are}"],
    ),
    "l2.rule2": (
        [
            "if something is {A_is} and {B_is}, then it is {C_is}",
            "every {A_sg} that is {B_is} is {C_is}",
            "all {A_pl} that are {B_are} are {C_are}",
        ],
        ["anything that is both {A_is} and {B_is} is {C_is}"],
    ),
    "l2.nrule2": (
        [
            "if something is {A_is} and {B_is}, then it is not {C_is}",
            "no {A_sg} that is {B_is} is {C_is}",
        ],
        ["anything that is both {A_is} and {B_is} is not {C_is}"],
    ),
    "l2.question": (
        ["is {c} {A_is}", "does it follow that {c} is {A_is}"],
        ["can we conclude that {c} is {A_is}"],
    ),
}


def _build_templates() -> Dict[str, Dict[str, List[Tuple[str, str]]]]:
    out = {}
    for construct, (train, heldout) in _TEMPLATE_TEXT.items():
        ids = [f"{construct}.{i}" for i in range(len(train) + len(heldout))]
        out[construct] = {
            "train": list(zip(ids[: len(train)], train)),
            "heldout": list(zip(ids[len(train) :], heldout)),
        }
    return out


TEMPLATES = _build_templates()

# #############################################################################
# Formulas: construction, rendering, parsing
# #############################################################################


def neg(f: Formula) -> Formula:
    """
    Negate a formula, removing a double negation (`neg(~p) == p`).
    """
    return f[1] if f[0] == "~" else ("~", f)


def _subst(f: Formula, var: str, const: str) -> Formula:
    """
    Replace the term `var` by `const` everywhere in `f`.
    """
    op = f[0]
    if op == "v":
        return f
    if op == "P":
        return ("P", f[1], const) if f[2] == var else f
    if op == "~":
        return ("~", _subst(f[1], var, const))
    if op in ("forall", "exists"):
        return f if f[1] == var else (op, f[1], _subst(f[2], var, const))
    return (op, _subst(f[1], var, const), _subst(f[2], var, const))


_PREC = {"v": 5, "P": 5, "forall": 5, "exists": 5, "~": 4, "&": 3, "|": 2, "->": 1}


def render_formula(f: Formula) -> str:
    """
    Render a formula with the minimal parentheses (`~` > `&` > `|` > `->`;
    `&`, `|` left-associative, `->` right-associative).
    """
    op = f[0]
    if op == "v":
        return f[1]
    if op == "P":
        return f"{f[1]}({f[2]})"
    if op in ("forall", "exists"):
        return f"{op} {f[1]} ({render_formula(f[2])})"
    if op == "~":
        s = render_formula(f[1])
        return "~" + (f"({s})" if _PREC[f[1][0]] < 4 else s)
    p = _PREC[op]
    left, right = render_formula(f[1]), render_formula(f[2])
    lp, rp = _PREC[f[1][0]], _PREC[f[2][0]]
    if lp < p or (lp == p and op == "->"):
        left = f"({left})"
    if rp < p or (rp == p and op != "->"):
        right = f"({right})"
    return f"{left} {op} {right}"


class ParseError(ValueError):
    """
    Raised when a string is not a well-formed symbolic formula.
    """


_TOKEN_RE = re.compile(r"\s*(->|=>|[~&|(),]|[A-Za-z_][A-Za-z0-9_]*)")
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*$")
_KEYWORDS = ("forall", "exists")


def _tokenize(s: str) -> List[str]:
    toks = []
    pos = 0
    s = s.rstrip()
    while pos < len(s):
        m = _TOKEN_RE.match(s, pos)
        if not m:
            raise ParseError(f"Bad character at {pos} in {s!r}")
        toks.append(m.group(1))
        pos = m.end()
    return toks


class _Parser:
    """
    Recursive-descent parser for the symbolic syntax.
    """

    def __init__(self, toks: List[str]) -> None:
        self.toks = toks
        self.i = 0

    def peek(self) -> Optional[str]:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def eat(self, tok: Optional[str] = None) -> str:
        t = self.peek()
        if t is None or (tok is not None and t != tok):
            raise ParseError(f"Expected {tok!r}, got {t!r}")
        self.i += 1
        return t

    def formula(self) -> Formula:
        left = self.disj()
        if self.peek() == "->":
            self.eat()
            return ("->", left, self.formula())
        return left

    def disj(self) -> Formula:
        f = self.conj()
        while self.peek() == "|":
            self.eat()
            f = ("|", f, self.conj())
        return f

    def conj(self) -> Formula:
        f = self.unary()
        while self.peek() == "&":
            self.eat()
            f = ("&", f, self.unary())
        return f

    def unary(self) -> Formula:
        t = self.peek()
        if t == "~":
            self.eat()
            return ("~", self.unary())
        if t == "(":
            self.eat()
            f = self.formula()
            self.eat(")")
            return f
        if t in _KEYWORDS:
            self.eat()
            var = self.eat()
            if not _IDENT_RE.match(var) or not var[0].islower() or var in _KEYWORDS:
                raise ParseError(f"Bad variable {var!r}")
            self.eat("(")
            body = self.formula()
            self.eat(")")
            return (t, var, body)
        if t is None or not _IDENT_RE.match(t):
            raise ParseError(f"Unexpected token {t!r}")
        self.eat()
        if t[0].isupper():
            self.eat("(")
            term = self.eat()
            if not _IDENT_RE.match(term) or not term[0].islower() or term in _KEYWORDS:
                raise ParseError(f"Bad term {term!r}")
            self.eat(")")
            return ("P", t, term)
        return ("v", t)


@functools.lru_cache(maxsize=100_000)
def parse_formula(s: str) -> Formula:
    """
    Parse one symbolic formula (e.g. `forall x (Cat(x) -> Mammal(x))`).

    :raise ParseError: if `s` is not a well-formed formula
    """
    p = _Parser(_tokenize(s))
    f = p.formula()
    if p.peek() is not None:
        raise ParseError(f"Trailing tokens in {s!r}")
    return f


def _parse_list(s: str) -> List[Formula]:
    """
    Parse a comma-separated list of formulas (the left side of a step).
    """
    p = _Parser(_tokenize(s))
    out = [p.formula()]
    while p.peek() == ",":
        p.eat()
        out.append(p.formula())
    if p.peek() is not None:
        raise ParseError(f"Trailing tokens in {s!r}")
    return out


def parse_premises(text: str) -> List[Formula]:
    """
    Parse a symbolic premise line (without "Premises: "), e.g. "p -> q. p.".
    """
    text = text.strip()
    if not text.endswith("."):
        raise ParseError("Premises must end with '.'")
    return [parse_formula(s) for s in text[:-1].split(". ")]


def parse_step(step: str) -> Tuple[List[Formula], Formula]:
    """
    Parse a symbolic step `<f1>, <f2> => <g>`.

    :return: (antecedent formulas, conclusion)
    :raise ParseError: if the step is not a symbolic step
    """
    parts = step.split("=>")
    if len(parts) != 2 or not parts[0].strip():
        raise ParseError(f"Not a step: {step!r}")
    return _parse_list(parts[0]), parse_formula(parts[1].strip())


def render_step(step: Step) -> str:
    """
    Render a step as `<f1>, <f2> => <g>`.
    """
    ante, concl, _ = step
    return ", ".join(render_formula(f) for f in ante) + " => " + render_formula(concl)


# #############################################################################
# z3
# #############################################################################

_OBJ = z3.DeclareSort("Obj")
_Z3_ATOMS: Dict[str, z3.BoolRef] = {}
_Z3_CONSTS: Dict[str, z3.ExprRef] = {}
_Z3_PREDS: Dict[str, z3.FuncDeclRef] = {}
_Z3_CACHE: Dict[Formula, z3.BoolRef] = {}


def _z3_const(name: str) -> z3.ExprRef:
    c = _Z3_CONSTS.get(name)
    if c is None:
        c = _Z3_CONSTS[name] = z3.Const(name, _OBJ)
    return c


def _to_z3(f: Formula, env: Optional[Dict[str, z3.ExprRef]] = None) -> z3.BoolRef:
    """
    Translate a formula: atoms are z3 Bools; L2 terms live in an
    uninterpreted sort `Obj` and predicates are functions `Obj -> Bool`.
    """
    if not env:
        hit = _Z3_CACHE.get(f)
        if hit is not None:
            return hit
    op = f[0]
    if op == "v":
        e = _Z3_ATOMS.get(f[1])
        if e is None:
            e = _Z3_ATOMS[f[1]] = z3.Bool(f[1])
    elif op == "P":
        fn = _Z3_PREDS.get(f[1])
        if fn is None:
            fn = _Z3_PREDS[f[1]] = z3.Function(f[1], _OBJ, z3.BoolSort())
        term = env[f[2]] if env and f[2] in env else _z3_const(f[2])
        e = fn(term)
    elif op == "~":
        e = z3.Not(_to_z3(f[1], env))
    elif op == "&":
        e = z3.And(_to_z3(f[1], env), _to_z3(f[2], env))
    elif op == "|":
        e = z3.Or(_to_z3(f[1], env), _to_z3(f[2], env))
    elif op == "->":
        e = z3.Implies(_to_z3(f[1], env), _to_z3(f[2], env))
    elif op in ("forall", "exists"):
        x = z3.Const(f"{f[1]}!b", _OBJ)
        body = _to_z3(f[2], {**(env or {}), f[1]: x})
        e = z3.ForAll([x], body) if op == "forall" else z3.Exists([x], body)
    else:
        raise ValueError(f"Unknown formula {f!r}")
    if not env:
        if len(_Z3_CACHE) > 200_000:
            _Z3_CACHE.clear()
        _Z3_CACHE[f] = e
    return e


_SOLVER: Optional[z3.Solver] = None


def _solver() -> z3.Solver:
    global _SOLVER
    if _SOLVER is None:
        _SOLVER = z3.Solver()
        _SOLVER.set("timeout", 5000)
    return _SOLVER


def z3_verdict(premises: Sequence[Formula], question: Formula) -> str:
    """
    Decide the verdict with z3.

    :return: "yes" if premises |= question, "no" if premises |= ~question,
        "unknown" if neither, "inconsistent" if the premises are
        unsatisfiable, "error" if z3 gives up
    """
    s = _solver()
    s.push()
    try:
        for f in premises:
            s.add(_to_z3(f))
        r = s.check()
        if r == z3.unsat:
            return "inconsistent"
        if r != z3.sat:
            return "error"
        q = _to_z3(question)
        res = []
        for goal in (z3.Not(q), q):
            s.push()
            s.add(goal)
            res.append(s.check())
            s.pop()
        if z3.unknown in res:
            return "error"
        if res[0] == z3.unsat:
            return "yes"
        if res[1] == z3.unsat:
            return "no"
        return "unknown"
    finally:
        s.pop()


def check_logic_step(step: str) -> Optional[bool]:
    """
    Check that a symbolic step's conclusion follows from its listed formulas.

    :param step: e.g. "p, p -> q => q" or
        "forall x (Cat(x) -> Mammal(x)), Cat(tom) => Mammal(tom)"
    :return: True / False (z3 entailment), or None if the step is not a
        parseable symbolic step (e.g. an English L3 step)
    """
    try:
        ante, concl = parse_step(step)
    except ParseError:
        return None
    s = _solver()
    s.push()
    try:
        for f in ante:
            s.add(_to_z3(f))
        s.add(z3.Not(_to_z3(concl)))
        r = s.check()
    finally:
        s.pop()
    if r == z3.unknown:
        return None
    return r == z3.unsat


# #############################################################################
# Forward chaining
# #############################################################################


@dataclasses.dataclass
class _Closure:
    """
    Result of forward chaining.

    :param known: formula -> index of the step deriving it (-1 = premise)
    :param steps: derivation steps in order
    :param consistent: False if a literal and its negation were derived
    """

    known: Dict[Formula, int]
    steps: List[Step]
    consistent: bool


def _is_literal(f: Formula) -> bool:
    return f[0] in ("v", "P") or (f[0] == "~" and f[1][0] in ("v", "P"))


def _conjuncts(body: Formula) -> Tuple[Formula, ...]:
    return (body[1], body[2]) if body[0] == "&" else (body,)


def _constants(fs: Sequence[Formula]) -> List[str]:
    out: Dict[str, None] = {}

    def walk(f: Formula, bound: Tuple[str, ...]) -> None:
        op = f[0]
        if op == "P":
            if f[2] not in bound:
                out[f[2]] = None
        elif op == "~":
            walk(f[1], bound)
        elif op in ("forall", "exists"):
            walk(f[2], bound + (f[1],))
        elif op != "v":
            walk(f[1], bound)
            walk(f[2], bound)

    for f in fs:
        walk(f, ())
    return list(out)


def _forward_chain(premises: Sequence[Formula], constants: Sequence[str]) -> _Closure:
    """
    Saturate the premises with MP, MT, and-elim, and-intro (for rule bodies)
    and disjunctive syllogism. L2 rules are instantiated on `constants`.
    """
    known: Dict[Formula, int] = {}
    for f in premises:
        known[f] = -1
    steps: List[Step] = []
    for f in premises:
        if _is_literal(f) and neg(f) in known:
            return _Closure(known, steps, False)
    rules = []
    conjs = []
    disjs = []
    for f in premises:
        op = f[0]
        if op == "->":
            rules.append((_conjuncts(f[1]), f[1], f[2], f, False))
        elif op == "forall":
            imp = f[2]
            for c in constants:
                b = _subst(imp[1], f[1], c)
                rules.append((_conjuncts(b), b, _subst(imp[2], f[1], c), f, True))
        elif op == "&":
            conjs.append(f)
        elif op == "|":
            disjs.append(f)

    def add(ante: Tuple[Formula, ...], concl: Formula, name: str) -> bool:
        known[concl] = len(steps)
        steps.append((ante, concl, name))
        return neg(concl) not in known

    changed = True
    while changed:
        changed = False
        for lits, body, head, src, fo in rules:
            if head not in known and all(b in known for b in lits):
                if len(lits) == 2 and body not in known:
                    add(lits, body, "and_intro")
                if not add((src, body) if fo else (body, src), head, "mp"):
                    return _Closure(known, steps, False)
                changed = True
            if len(lits) == 1:
                nb, nh = neg(body), neg(head)
                if nb not in known and nh in known:
                    if not add((src, nh) if fo else (nh, src), nb, "mt"):
                        return _Closure(known, steps, False)
                    changed = True
        for c in conjs:
            for part in (c[1], c[2]):
                if part not in known:
                    if not add((c,), part, "and_elim"):
                        return _Closure(known, steps, False)
                    changed = True
        for dj in disjs:
            a, b = dj[1], dj[2]
            for x, y in ((a, b), (b, a)):
                if y not in known and neg(x) in known:
                    if not add((dj, neg(x)), y, "ds"):
                        return _Closure(known, steps, False)
                    changed = True
    return _Closure(known, steps, True)


def _proof(cl: _Closure, target: Formula) -> List[Step]:
    """
    Extract the steps needed to derive `target`, in derivation order.
    """
    need: Set[int] = set()
    stack = [target]
    while stack:
        i = cl.known[stack.pop()]
        if i >= 0 and i not in need:
            need.add(i)
            stack.extend(cl.steps[i][0])
    return [cl.steps[i] for i in sorted(need)]


def _fc_solution(cl: _Closure, question: Formula) -> Tuple[str, List[Step]]:
    """
    Forward-chaining verdict and the steps shown in the solution.
    """
    if question in cl.known:
        return "yes", _proof(cl, question)
    if neg(question) in cl.known:
        return "no", _proof(cl, neg(question))
    return "unknown", list(cl.steps)


# #############################################################################
# Symbolic problem generation (L1 / L2)
# #############################################################################


@dataclasses.dataclass
class _Problem:
    """
    A verified symbolic problem.
    """

    level: str
    premises: List[Formula]
    question: Formula
    verdict: str
    steps: List[Step]
    meta: Dict[str, object]


class _Names:
    """
    Hands out fresh names from a per-example shuffled pool.
    """

    def __init__(self, pool: List[str]) -> None:
        self.pool = pool
        self.i = 0

    def new(self) -> str:
        if self.i >= len(self.pool):
            raise _Fail("name pool exhausted")
        self.i += 1
        return self.pool[self.i - 1]

    def peek(self) -> Optional[str]:
        return self.pool[self.i] if self.i < len(self.pool) else None


class _Fail(Exception):
    """
    Internal: this attempt failed, retry with fresh randomness.
    """


def _choose(rng: random.Random, weights: Dict[str, float]) -> str:
    keys = [k for k, w in weights.items() if w > 0]
    return rng.choices(keys, [weights[k] for k in keys])[0]


def _l1_chain(rng: random.Random, names: _Names, tgt: Formula, d: int) -> List[Formula]:
    """
    Build premises from which `tgt` is derived in exactly `d` steps (all
    atoms fresh, so the derivation is unique).
    """

    def lit() -> Formula:
        a = ("v", names.new())
        return a if rng.random() < 0.75 else ("~", a)

    premises: List[Formula] = []
    goals = [tgt]
    remaining = d
    while remaining > 0:
        i = rng.randrange(len(goals))
        g = goals[i]
        op = _choose(
            rng,
            {
                "mp": 4.0,
                "mp2": 2.0 if remaining >= 2 else 0.0,
                "mt": 2.0 if g[0] == "~" else 0.5,
                "ds": 1.5,
                "ae": 1.0 if (remaining == 1 or len(goals) > 1) else 0.0,
            },
        )
        if op == "mp":
            b = lit()
            premises.append(("->", b, g))
            goals[i] = b
            remaining -= 1
        elif op == "mp2":
            b1, b2 = lit(), lit()
            premises.append(("->", ("&", b1, b2), g))
            goals[i : i + 1] = [b1, b2]
            remaining -= 2
        elif op == "mt":
            h = lit()
            premises.append(("->", neg(g), h))
            goals[i] = neg(h)
            remaining -= 1
        elif op == "ds":
            a = lit()
            premises.append(("|", a, g) if rng.random() < 0.5 else ("|", g, a))
            goals[i] = neg(a)
            remaining -= 1
        else:
            b = lit()
            premises.append(("&", g, b) if rng.random() < 0.5 else ("&", b, g))
            goals.pop(i)
            remaining -= 1
    return premises + goals


def _l2_chain(rng: random.Random, names: _Names, tgt: Formula, d: int) -> List[Formula]:
    """
    L2 version of `_l1_chain`: all steps are about the constant of `tgt`.
    """
    c = (tgt[1] if tgt[0] == "~" else tgt)[2]

    def lift(g: Formula) -> Formula:
        return ("~", ("P", g[1][1], VAR)) if g[0] == "~" else ("P", g[1], VAR)

    premises: List[Formula] = []
    goals = [tgt]
    remaining = d
    while remaining > 0:
        i = rng.randrange(len(goals))
        g = goals[i]
        op = _choose(
            rng,
            {
                "mp": 4.0,
                "mp2": 2.0 if remaining >= 2 else 0.0,
                "mt": 3.0 if g[0] == "~" else 0.0,
            },
        )
        if op == "mp":
            a = names.new()
            premises.append(("forall", VAR, ("->", ("P", a, VAR), lift(g))))
            goals[i] = ("P", a, c)
            remaining -= 1
        elif op == "mp2":
            a, b = names.new(), names.new()
            body = ("&", ("P", a, VAR), ("P", b, VAR))
            premises.append(("forall", VAR, ("->", body, lift(g))))
            goals[i : i + 1] = [("P", a, c), ("P", b, c)]
            remaining -= 2
        else:
            h = names.new()
            head = ("P", h, VAR) if rng.random() < 0.6 else ("~", ("P", h, VAR))
            premises.append(("forall", VAR, ("->", lift(neg(g)), head)))
            goals[i] = neg(_subst(head, VAR, c))
            remaining -= 1
    return premises + goals


def _known_lits(premises: List[Formula], constants: Sequence[str]) -> List[Formula]:
    cl = _forward_chain(premises, constants)
    return [f for f in cl.known if _is_literal(f)]


def _l1_trap(rng: random.Random, names: _Names, premises: List[Formula]) -> Tuple[List[Formula], Formula, str]:
    """
    Add a premise that mentions the question atom without settling it.
    """
    known = _known_lits(premises, ())
    if not known:
        raise _Fail("no known literal")
    lit_ = rng.choice(known)
    kind = _choose(rng, {"converse": 1.0, "disj": 1.0, "conj": 1.0, "deny": 1.0})
    h = ("v", names.new())
    if kind == "converse":
        body = h if rng.random() < 0.75 else ("~", h)
        return [("->", body, lit_)], h, kind
    if kind == "disj":
        hl = h if rng.random() < 0.75 else ("~", h)
        return [("|", lit_, hl) if rng.random() < 0.5 else ("|", hl, lit_)], h, kind
    if kind == "conj":
        z = ("v", names.new())
        hl = h if rng.random() < 0.75 else ("~", h)
        zl = z if rng.random() < 0.7 else ("~", z)
        body = ("&", lit_, hl) if rng.random() < 0.5 else ("&", hl, lit_)
        return [("->", body, zl)], (z if rng.random() < 0.7 else h), kind
    hl = h if rng.random() < 0.7 else ("~", h)
    return [("->", neg(lit_), hl)], h, kind


def _l2_trap(
    rng: random.Random, names: _Names, premises: List[Formula], consts: List[str]
) -> Tuple[List[Formula], Formula, str]:
    """
    L2 version of `_l1_trap`.
    """
    known = _known_lits(premises, consts[:1])
    pos = [f for f in known if f[0] == "P"]
    negs = [f for f in known if f[0] == "~"]
    kind = _choose(
        rng,
        {"converse": 1.0, "conj": 1.0 if pos else 0.0, "deny": 1.0 if negs else 0.0, "other": 2.0},
    )
    c0 = consts[0]
    if kind == "converse":
        lit_ = rng.choice(known)
        h = names.new()
        head = ("~", ("P", lit_[1][1], VAR)) if lit_[0] == "~" else ("P", lit_[1], VAR)
        return [("forall", VAR, ("->", ("P", h, VAR), head))], ("P", h, c0), kind
    if kind == "conj":
        lit_ = rng.choice(pos)
        h, z = names.new(), names.new()
        parts = [("P", lit_[1], VAR), ("P", h, VAR)]
        rng.shuffle(parts)
        head = ("P", z, VAR) if rng.random() < 0.6 else ("~", ("P", z, VAR))
        q = ("P", z, c0) if rng.random() < 0.7 else ("P", h, c0)
        return [("forall", VAR, ("->", ("&", parts[0], parts[1]), head))], q, kind
    if kind == "deny":
        lit_ = rng.choice(negs)
        z = names.new()
        head = ("P", z, VAR) if rng.random() < 0.6 else ("~", ("P", z, VAR))
        return [("forall", VAR, ("->", ("P", lit_[1][1], VAR), head))], ("P", z, c0), kind
    # "other": a fact about a second constant, question about it.
    c1 = consts[1]
    preds = sorted({(f[1] if f[0] == "~" else f)[1] for f in known})
    a, b = rng.choice(preds), rng.choice(preds)
    fact = ("P", a, c1) if rng.random() < 0.7 else ("~", ("P", a, c1))
    return [fact], ("P", b, c1), kind


def _l1_distractor(rng: random.Random, names: _Names, premises: List[Formula], q: Formula) -> Formula:
    used = sorted(_atoms(premises))
    fresh = names.peek() if names.i < len(ATOMS) else None

    def lit() -> Formula:
        r = rng.random()
        if r < 0.2:
            a = q
        elif r < 0.4 and fresh is not None:
            a = ("v", fresh)
        else:
            a = ("v", rng.choice(used))
        return a if rng.random() < 0.7 else ("~", a)

    kind = _choose(rng, {"fact": 2.0, "rule": 4.0, "rule2": 1.0, "or": 1.5, "and": 1.0})
    if kind == "fact":
        return lit()
    if kind == "rule":
        return ("->", lit(), lit())
    if kind == "rule2":
        return ("->", ("&", lit(), lit()), lit())
    return ("|" if kind == "or" else "&", lit(), lit())


def _l2_distractor(
    rng: random.Random, names: _Names, premises: List[Formula], q: Formula, consts: List[str]
) -> Formula:
    used = sorted(_preds(premises))
    fresh = names.peek()

    def pred() -> str:
        r = rng.random()
        if r < 0.2:
            return q[1]
        if r < 0.45 and fresh is not None:
            return fresh
        return rng.choice(used)

    def hd(p: str) -> Formula:
        return ("P", p, VAR) if rng.random() < 0.65 else ("~", ("P", p, VAR))

    kind = _choose(rng, {"fact": 4.0, "rule": 3.0, "rule2": 1.0})
    if kind == "fact":
        # Mostly about other constants, so several individuals appear.
        c = consts[0] if rng.random() < 0.3 else rng.choice(consts[1:])
        f = ("P", pred(), c)
        return f if rng.random() < 0.7 else ("~", f)
    if kind == "rule":
        return ("forall", VAR, ("->", ("P", pred(), VAR), hd(pred())))
    body = ("&", ("P", pred(), VAR), ("P", pred(), VAR))
    return ("forall", VAR, ("->", body, hd(pred())))


def _atoms(fs: Sequence[Formula]) -> Set[str]:
    out: Set[str] = set()

    def walk(f: Formula) -> None:
        if f[0] == "v":
            out.add(f[1])
        elif f[0] == "~":
            walk(f[1])
        elif f[0] in ("&", "|", "->"):
            walk(f[1])
            walk(f[2])

    for f in fs:
        walk(f)
    return out


def _preds(fs: Sequence[Formula]) -> Set[str]:
    out: Set[str] = set()

    def walk(f: Formula) -> None:
        op = f[0]
        if op == "P":
            out.add(f[1])
        elif op == "~":
            walk(f[1])
        elif op in ("forall", "exists"):
            walk(f[2])
        elif op != "v":
            walk(f[1])
            walk(f[2])

    for f in fs:
        walk(f)
    return out


def _has_neg(f: Formula) -> bool:
    if f[0] == "~":
        return True
    if f[0] in ("v", "P"):
        return False
    return any(_has_neg(g) for g in f[1:] if isinstance(g, tuple))


def _is_degenerate(f: Formula) -> bool:
    """
    Reject distractors like `p -> p`, `p | ~p`, `p & p`.
    """
    if f[0] == "forall":
        f = f[2]
    if f[0] in ("&", "|", "->"):
        lits = list(_conjuncts(f[1])) + [f[2]] if f[0] == "->" else [f[1], f[2]]
        atoms = [(x[1] if x[0] == "~" else x) for x in lits]
        return len(set(atoms)) < len(atoms)
    return False


def _gen_symbolic(
    rng: random.Random,
    level: str,
    verdict: str,
    depth: int,
    min_depth: int,
    max_depth: int,
    max_premises: int,
) -> _Problem:
    """
    One attempt at a verified L1 / L2 problem.

    :raise _Fail: if the attempt does not meet the constraints
    """
    l1 = level == "L1"
    if l1:
        base = list(ATOMS)
        rng.shuffle(base)
        names = _Names(base + list(EXTRA_ATOMS))
        consts: List[str] = []
    else:
        preds = list(PREDICATES)
        rng.shuffle(preds)
        names = _Names(preds)
        consts = rng.sample(CONSTANTS, rng.randint(3, 5))
    chain = _l1_chain if l1 else _l2_chain
    trap_kind = None
    if verdict == "unknown":
        g = ("v", names.new()) if l1 else ("P", names.new(), consts[0])
        if rng.random() < 0.5:
            g = ("~", g)
        premises = chain(rng, names, g, depth)
        trap, q, trap_kind = (
            _l1_trap(rng, names, premises) if l1 else _l2_trap(rng, names, premises, consts)
        )
        premises += trap
    else:
        q = ("v", names.new()) if l1 else ("P", names.new(), consts[0])
        premises = chain(rng, names, q if verdict == "yes" else ("~", q), depth)
    if len(premises) > max_premises or len(set(premises)) < len(premises):
        raise _Fail("too many premises")
    rng.shuffle(premises)

    def evaluate(prems: List[Formula]) -> Optional[List[Step]]:
        cl = _forward_chain(prems, consts if not l1 else ())
        if not cl.consistent:
            return None
        v, steps = _fc_solution(cl, q)
        if v != verdict:
            return None
        # yes / no: the proof has exactly `depth` steps (no shortcut through
        # a distractor); unknown: the closure has depth..max_depth steps.
        if verdict != "unknown" and len(steps) != depth:
            return None
        if verdict == "unknown" and not (max(depth, min_depth) <= len(steps) <= max_depth):
            return None
        return steps

    steps = evaluate(premises)
    if steps is None:
        raise _Fail("chain does not meet constraints")
    n_chain = len(premises)
    # The total premise count is drawn independently of the verdict, so it
    # is not a cue for the label.
    total = rng.randint(min(depth + 2, max_premises), max_premises)
    target = max(0, total - n_chain)
    # `no` problems always contain a negation; make it a weak cue for the
    # other verdicts too by usually adding a negated distractor.
    force_neg = not any(_has_neg(f) for f in premises) and rng.random() < 0.9
    if force_neg:
        target = max(target, 1)
    if n_chain + target > max_premises:
        raise _Fail("no room for distractors")
    tries = 0
    while len(premises) < n_chain + target and tries < 3 * target + 3:
        tries += 1
        cand = (
            _l1_distractor(rng, names, premises, q)
            if l1
            else _l2_distractor(rng, names, premises, q, consts)
        )
        if cand in premises or neg(cand) in premises or _is_degenerate(cand):
            continue
        if force_neg and not _has_neg(cand) and len(premises) == n_chain + target - 1:
            continue
        trial = list(premises)
        trial.insert(rng.randint(0, len(trial)), cand)
        new_steps = evaluate(trial)
        if new_steps is None:
            continue
        premises, steps = trial, new_steps
        nxt = names.peek()
        if nxt is not None and nxt in (_atoms([cand]) if l1 else _preds([cand])):
            names.new()
    if force_neg and not any(_has_neg(f) for f in premises):
        raise _Fail("no negated distractor")
    z3v = z3_verdict(premises, q)
    if z3v != verdict:
        raise _Fail(f"z3 says {z3v}, forward chaining says {verdict}")
    meta: Dict[str, object] = {
        "level": level,
        "source": level,
        "verdict": verdict,
        "depth": len(steps),
        "n_premises": len(premises),
        "n_distractors": len(premises) - n_chain,
        "rules": sorted({s[2] for s in steps}),
    }
    if l1:
        meta["n_atoms"] = len(_atoms(premises + [q]))
    else:
        meta["n_constants"] = len(_constants(premises + [q]))
        meta["n_predicates"] = len(_preds(premises + [q]))
    if trap_kind:
        meta["trap"] = trap_kind
    return _Problem(level, premises, q, verdict, steps, meta)


# #############################################################################
# English rendering (L3)
# #############################################################################


def _cap(s: str) -> str:
    return s[0].upper() + s[1:]


class _English:
    """
    Renders the formulas of one symbolic problem in English.
    """

    def __init__(self, prob: _Problem, rng: random.Random, split: str) -> None:
        self.rng = rng
        self.split = split
        self.used_ids: Set[str] = set()
        self.l1 = prob.level == "L1"
        if self.l1:
            atoms = sorted(_atoms(prob.premises + [prob.question]))
            self.props = dict(zip(atoms, rng.sample(PROPOSITIONS, len(atoms))))

    def _pick(self, construct: str) -> str:
        tid, fmt = self.rng.choice(TEMPLATES[construct][self.split])
        self.used_ids.add(tid)
        return fmt

    def lit(self, f: Formula) -> str:
        """
        Plain clause for a literal ("it rains", "Tom is not a cat").
        """
        if f[0] == "~":
            g = f[1]
            if g[0] == "v":
                return self.props[g[1]][1]
            return f"{_cap(g[2])} is not {LEXICON[g[1]].is_sg}"
        if f[0] == "v":
            return self.props[f[1]][0]
        return f"{_cap(f[2])} is {LEXICON[f[1]].is_sg}"

    def plain(self, f: Formula) -> str:
        """
        Plain clause for a derived formula (literal or conjunction).
        """
        if f[0] == "&":
            return f"{self.plain(f[1])} and {self.plain(f[2])}"
        return self.lit(f)

    def premise(self, f: Formula) -> str:
        """
        Clause for a premise using a randomly chosen template.
        """
        op = f[0]
        if op == "->":
            body, head = f[1], f[2]
            if body[0] == "&":
                return self._pick("l1.rule2").format(
                    a=self.lit(body[1]), b=self.lit(body[2]), c=self.lit(head)
                )
            return self._pick("l1.rule").format(a=self.lit(body), b=self.lit(head))
        if op == "&":
            return self._pick("l1.and").format(a=self.lit(f[1]), b=self.lit(f[2]))
        if op == "|":
            return self._pick("l1.or").format(a=self.lit(f[1]), b=self.lit(f[2]))
        if op == "forall":
            body, head = f[2][1], f[2][2]
            negated = head[0] == "~"
            hp = head[1][1] if negated else head[1]
            # Body predicates then head predicate -> A, B (, C).
            kw: Dict[str, str] = {}
            preds = [b[1] for b in _conjuncts(body)] + [hp]
            for letter, p in zip("ABC", preds):
                w = LEXICON[p]
                kw.update({f"{letter}_is": w.is_sg, f"{letter}_are": w.is_pl})
                kw.update({f"{letter}_sg": w.sg, f"{letter}_pl": w.pl})
            if body[0] == "&":
                construct = "l2.nrule2" if negated else "l2.rule2"
            else:
                construct = "l2.nrule" if negated else "l2.rule"
            return self._pick(construct).format(**kw)
        return self._pick("fact").format(l=self.lit(f))

    def question(self, q: Formula) -> str:
        if q[0] == "v":
            return self._pick("l1.question").format(a=self.props[q[1]][0])
        return self._pick("l2.question").format(c=_cap(q[2]), A_is=LEXICON[q[1]].is_sg)


def _english_example(prob: _Problem, rng: random.Random, split: str) -> lsform.Example:
    en = _English(prob, rng, split)
    clauses = {f: en.premise(f) for f in prob.premises}
    problem = " ".join(_cap(clauses[f]) + "." for f in prob.premises)
    question = _cap(en.question(prob.question))
    steps = []
    for ante, concl, _ in prob.steps:
        # Facts before rules ("Tom is a cat, all cats are mammals => ...").
        ordered = sorted(ante, key=lambda f: f[0] in ("forall", "->", "|"))
        parts = [clauses[f] if f in clauses else en.plain(f) for f in ordered]
        steps.append(", ".join(parts) + " => " + en.plain(concl))
    meta = dict(prob.meta)
    meta.update(
        {
            "level": "L3",
            "templates": sorted(en.used_ids),
            "template_split": split,
            "sym_premises": ". ".join(render_formula(f) for f in prob.premises) + ".",
            "sym_question": render_formula(prob.question),
        }
    )
    return lsform.Example(
        task="logic", problem=problem, question=question, steps=steps,
        answer=prob.verdict, meta=meta,
    )  # fmt: skip


# #############################################################################
# Public API
# #############################################################################


def generate_example(
    rng: random.Random,
    level: str = "L1",
    *,
    verdict: Optional[str] = None,
    depth: Optional[int] = None,
    min_depth: int = 1,
    max_depth: int = 4,
    max_premises: Optional[int] = None,
    template_split: str = "train",
    source: Optional[str] = None,
    max_attempts: int = 500,
) -> lsform.Example:
    """
    Generate one verified logic example.

    :param rng: random generator (controls reproducibility)
    :param level: "L1" (propositional), "L2" (first-order) or "L3" (English)
    :param verdict: target verdict ("yes" / "no" / "unknown"); random if None
    :param depth: target number of solution steps; uniform in
        [min_depth, max_depth] if None. `yes` / `no` need depth >= 1; for
        `unknown` it is the length of the chain before the trap premise
    :param min_depth: minimum number of solution steps (all verdicts)
    :param max_depth: maximum number of solution steps (all verdicts)
    :param max_premises: hard cap on the number of premises; default
        `max(5, depth + 2)` (room for a depth-`depth` chain, its fact, a
        trap premise and some distractors)
    :param template_split: L3 only, "train" or "heldout" template set
    :param source: L3 only, symbolic source "L1" / "L2" (random if None)
    :param max_attempts: attempts before giving up
    :return: logic example (meta: level, verdict, depth, n_premises, ...)
    """
    if level not in LEVELS:
        raise ValueError(f"Unknown level '{level}'")
    if template_split not in TEMPLATE_SPLITS:
        raise ValueError(f"Unknown template split '{template_split}'")
    if not 0 <= min_depth <= max_depth:
        raise ValueError("Need 0 <= min_depth <= max_depth")
    if verdict is not None and verdict not in VERDICTS:
        raise ValueError(f"Unknown verdict '{verdict}'")
    if verdict is None:
        verdict = rng.choice(VERDICTS)
    if verdict != "unknown" and max_depth < 1:
        raise ValueError("yes / no examples need max_depth >= 1")
    lo = min_depth if verdict == "unknown" else max(1, min_depth)
    if depth is None:
        depth = rng.randint(lo, max_depth)
    if not lo <= depth <= max_depth:
        raise ValueError(f"depth {depth} outside [{lo}, {max_depth}]")
    cap = max_premises if max_premises is not None else max(5, depth + 2)
    sym_level = level
    if level == "L3":
        sym_level = source or rng.choice(("L1", "L2"))
    for _ in range(max_attempts):
        try:
            prob = _gen_symbolic(rng, sym_level, verdict, depth, lo, max_depth, cap)
        except _Fail:
            continue
        if level == "L3":
            return _english_example(prob, rng, template_split)
        return lsform.Example(
            task="logic",
            problem=". ".join(render_formula(f) for f in prob.premises) + ".",
            question=render_formula(prob.question),
            steps=[render_step(s) for s in prob.steps],
            answer=prob.verdict,
            meta=prob.meta,
        )
    raise RuntimeError(
        f"No {level} '{verdict}' example with depth {depth} and <= {cap} premises "
        f"after {max_attempts} attempts; relax max_premises"
    )


def example_key(ex: lsform.Example) -> str:
    """
    Deduplication key of an example: its full prompt (premises + question).
    """
    return lsform.render_prompt(ex)


def generate_dataset(
    n: int,
    seed: int,
    *,
    level: str = "L1",
    depth_range: Tuple[int, int] = (1, 4),
    max_premises: Optional[int] = None,
    template_split: str = "train",
    exclude: Optional[set] = None,
) -> Iterator[lsform.Example]:
    """
    Yield `n` unique examples with verdicts balanced (each block of three
    holds one yes / no / unknown in random order) and depth uniform in
    `depth_range`.

    :param exclude: keys (`example_key` / `problem_from_record`) to skip,
        e.g. a held-out test set
    """
    rng = random.Random(seed)
    seen = set(exclude or ())
    made = 0
    attempts = 0
    block: List[str] = []
    while made < n:
        if not block:
            block = list(VERDICTS)
            rng.shuffle(block)
        verdict = block[-1]
        attempts += 1
        if attempts > 50 * n + 1000:
            raise RuntimeError(f"Only {made}/{n} unique examples; widen the ranges")
        lo = depth_range[0] if verdict == "unknown" else max(1, depth_range[0])
        ex = generate_example(
            rng,
            level,
            verdict=verdict,
            depth=rng.randint(lo, depth_range[1]),
            min_depth=depth_range[0],
            max_depth=depth_range[1],
            max_premises=max_premises,
            template_split=template_split,
        )
        key = example_key(ex)
        if key in seen:
            continue
        seen.add(key)
        block.pop()
        made += 1
        yield ex


def example_to_record(ex: lsform.Example) -> dict:
    """
    Serialize an example to a JSONL record (rendered text + fields).
    """
    return {
        "format": lsform.FORMAT_VERSION,
        "task": ex.task,
        "prompt": lsform.render_prompt(ex),
        "solution": lsform.render_solution(ex),
        "answer": ex.answer,
        **ex.meta,
    }


def problem_from_record(record: dict) -> str:
    """
    Recover the deduplication key (`example_key`) from a JSONL record.
    """
    prompt = record["prompt"]
    if not prompt.startswith("Premises: "):
        raise ValueError(f"Not a logic record: {prompt[:40]!r}")
    return prompt


def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--level", choices=LEVELS, default="L1")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-depth", type=int, default=1)
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--max-premises", type=int, default=None)
    ap.add_argument("--template-split", choices=TEMPLATE_SPLITS, default="train")
    ap.add_argument("--out", required=True, help="output .jsonl path")
    ap.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="JSONL files whose prompts must not be generated (test sets)",
    )
    a = ap.parse_args()
    exclude = set()
    for path in a.exclude:
        with open(path) as f:
            exclude.update(problem_from_record(json.loads(line)) for line in f)
    with open(a.out, "w") as f:
        for ex in generate_dataset(
            a.n,
            a.seed,
            level=a.level,
            depth_range=(a.min_depth, a.max_depth),
            max_premises=a.max_premises,
            template_split=a.template_split,
            exclude=exclude,
        ):
            f.write(json.dumps(example_to_record(ex)) + "\n")


if __name__ == "__main__":
    _main()
