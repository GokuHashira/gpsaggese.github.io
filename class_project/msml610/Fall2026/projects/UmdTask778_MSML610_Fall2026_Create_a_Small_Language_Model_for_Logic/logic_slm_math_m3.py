"""
M3 generator: word problems rendered from a computation graph.

A problem is a k-step computation graph over typed quantities (item counts,
container counts, money, unit prices). Each graph node applies one operation
to an earlier quantity and either a number given in its sentence or another
quantity. Two graph shapes are sampled:

- chain: a start quantity followed by k operations;
- join: two independent chains (two people, or a count and a unit price)
  joined by a final node (total, difference, or count * price).

Each node is rendered with a randomly chosen sentence template; names, items,
containers, units and numbers are randomized, irrelevant distractor
sentences can be inserted, and the question about the final node comes last.
The solution has one step per node, `<short description>: a op b = c`, in
topological order (see `references/format_v1.md`, section 2). All values are
non-negative integers and every division is exact.

Every template has an id; about a quarter of the templates of each group are
`heldout` and only used with `template_split="heldout"`, so whole templates
can be kept out of training.

Import as:

import logic_slm_math_m3 as lsmam3
"""

import argparse
import dataclasses
import json
import random
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import logic_slm_format as lsform
import logic_slm_math_m1 as lsmam1

TEMPLATE_SPLITS = ("train", "heldout", "all")

# #############################################################################
# Entities
# #############################################################################

_NAMES_F = (
    "Sarah", "Emma", "Olivia", "Ava", "Mia", "Sofia", "Lily", "Chloe", "Grace",
    "Zoe", "Hannah", "Nora", "Aisha", "Priya", "Mei", "Yuki", "Fatima",
    "Elena", "Isabel", "Rosa", "Leila", "Amara", "Ingrid", "Chen", "Maya",
    "Julia", "Anna", "Clara", "Ruth", "Tara", "Nadia", "Keiko", "Lucia",
    "Bianca", "Freya", "Sana", "Ines", "Diana", "Paula", "Wendy",
)
_NAMES_M = (
    "Tom", "James", "Liam", "Noah", "Ethan", "Lucas", "Mason", "Leo", "Omar",
    "Ravi", "Kenji", "Diego", "Mateo", "Ali", "Samuel", "Daniel", "Hugo",
    "Felix", "Arjun", "Wei", "Kwame", "Ivan", "Pablo", "Marco", "Tariq",
    "Jonas", "Oscar", "Henry", "Jack", "Ben", "Luis", "Malik", "Hassan",
    "Nikolai", "Sean", "Victor", "Theo", "Andre", "Carlos", "Dev",
)
_PRONOUNS = {
    "f": {"subj": "she", "obj": "her", "pos": "her", "self": "herself"},
    "m": {"subj": "he", "obj": "him", "pos": "his", "self": "himself"},
}


@dataclasses.dataclass(frozen=True)
class _Item:
    """
    A kind of thing that is counted, with its container.

    :param tags: "food", "measure" (a unit of measure such as liters),
        "fragile"
    """

    plural: str
    singular: str
    container: str
    containers: str
    price: Tuple[int, int]
    tags: Tuple[str, ...] = ()


def _it(p: str, s: str, c: str, cs: str, lo: int, hi: int, *tags: str) -> _Item:
    return _Item(p, s, c, cs, (lo, hi), tags)


_ITEMS = (
    _it("apples", "apple", "basket", "baskets", 1, 3, "food"),
    _it("eggs", "egg", "tray", "trays", 1, 2, "food", "fragile"),
    _it("oranges", "orange", "bag", "bags", 1, 3, "food"),
    _it("cookies", "cookie", "jar", "jars", 1, 4, "food"),
    _it("cupcakes", "cupcake", "box", "boxes", 2, 5, "food"),
    _it("muffins", "muffin", "tray", "trays", 2, 4, "food"),
    _it("donuts", "donut", "box", "boxes", 1, 3, "food"),
    _it("tomatoes", "tomato", "crate", "crates", 1, 2, "food"),
    _it("potatoes", "potato", "sack", "sacks", 1, 2, "food"),
    _it("bananas", "banana", "bunch", "bunches", 1, 2, "food"),
    _it("lemons", "lemon", "bag", "bags", 1, 2, "food"),
    _it("peaches", "peach", "crate", "crates", 1, 3, "food"),
    _it("pies", "pie", "rack", "racks", 8, 20, "food"),
    _it("sandwiches", "sandwich", "platter", "platters", 4, 9, "food"),
    _it("pencils", "pencil", "box", "boxes", 1, 2),
    _it("crayons", "crayon", "box", "boxes", 1, 2),
    _it("notebooks", "notebook", "pack", "packs", 2, 6),
    _it("stickers", "sticker", "sheet", "sheets", 1, 2),
    _it("marbles", "marble", "bag", "bags", 1, 2, "fragile"),
    _it("beads", "bead", "jar", "jars", 1, 2),
    _it("buttons", "button", "jar", "jars", 1, 2),
    _it("stamps", "stamp", "album", "albums", 1, 5),
    _it("trading cards", "trading card", "pack", "packs", 1, 4),
    _it("comic books", "comic book", "stack", "stacks", 3, 9),
    _it("books", "book", "shelf", "shelves", 5, 20),
    _it("candles", "candle", "box", "boxes", 2, 8),
    _it("seashells", "seashell", "bucket", "buckets", 1, 3, "fragile"),
    _it("roses", "rose", "bouquet", "bouquets", 2, 5),
    _it("tulips", "tulip", "vase", "vases", 1, 4),
    _it("toy cars", "toy car", "box", "boxes", 3, 10),
    _it("batteries", "battery", "pack", "packs", 1, 3),
    _it("balloons", "balloon", "bag", "bags", 1, 2),
    _it("light bulbs", "light bulb", "carton", "cartons", 2, 6, "fragile"),
    _it("tiles", "tile", "box", "boxes", 2, 7, "fragile"),
    _it("plants", "plant", "row", "rows", 3, 12),
    _it("chairs", "chair", "row", "rows", 10, 40),
    _it("glasses", "glass", "case", "cases", 2, 6, "fragile"),
    _it("bottles of water", "bottle of water", "case", "cases", 1, 2),
    _it("cans of soup", "can of soup", "carton", "cartons", 1, 3, "food"),
    _it("liters of juice", "liter of juice", "jug", "jugs", 1, 4, "measure", "food"),
    _it("meters of rope", "meter of rope", "spool", "spools", 1, 3, "measure"),
    _it("pounds of flour", "pound of flour", "bag", "bags", 1, 2, "measure", "food"),
    _it("kilograms of rice", "kilogram of rice", "sack", "sacks", 1, 3, "measure", "food"),
    _it("gallons of paint", "gallon of paint", "can", "cans", 10, 30, "measure"),
    _it("feet of ribbon", "foot of ribbon", "roll", "rolls", 1, 2, "measure"),
    _it("cups of sugar", "cup of sugar", "bag", "bags", 1, 2, "measure", "food"),
)
_THINGS = (
    "lunch", "a movie ticket", "a new hat", "bus fare", "a birthday gift",
    "snacks", "a phone case", "school supplies", "a pizza", "a scarf",
    "a museum ticket", "a board game", "a backpack", "groceries", "a haircut",
    "a water bottle", "a pair of gloves", "a puzzle", "train tickets", "a plant",
)
_JOBS = (
    "walking dogs", "mowing lawns", "babysitting", "tutoring", "washing cars",
    "a lemonade stand", "selling drawings", "raking leaves", "a bake sale",
    "delivering newspapers", "fixing bikes", "pet sitting",
)
_PLACES = (
    "the library", "the park", "the beach", "school", "the market",
    "the zoo", "the gym", "the museum", "the farm", "the station",
)
_RELATIVES = ("brother", "sister", "cousin", "aunt", "uncle", "grandmother", "neighbor")
_PETS = ("cats", "dogs", "goldfish", "rabbits", "parrots", "hamsters", "chickens")

# #############################################################################
# Templates
# #############################################################################


@dataclasses.dataclass(frozen=True)
class _Template:
    """
    A sentence template for one graph node (or a start / question sentence).

    :param tid: unique id; templates of one group share the prefix before
        the last "_"
    :param src: semantic type of the input quantity ("" for starts)
    :param dst: semantic type of the output quantity
    :param op: operation (`+ - * /`), "" for start sentences
    :param num: how the given number is sampled: "amount", "small", "mult",
        "div", "price", "money", "sub", "start"
    :param text: sentence; slots `{H}` holder, `{n}` number, `{items}`,
        `{item}`, `{container}`, `{containers}`, `{stuff}` (the
        material, e.g. "ribbon" for "feet of ribbon"), pronouns `{subj}` `{obj}`
        `{pos}` `{self}`, `{N2}` another name, `{thing}`, `{job}`, `{items2}`
    :param desc: step description (no ":")
    :param tags_no: item tags that rule the template out
    :param tags_need: item tags required by the template
    """

    tid: str
    src: str
    dst: str
    op: str
    num: str
    text: str
    desc: str
    tags_no: Tuple[str, ...] = ()
    tags_need: Tuple[str, ...] = ()
    heldout: bool = False


def _group(
    prefix: str,
    src: str,
    dst: str,
    op: str,
    num: str,
    rows: Sequence[Tuple],
) -> List[_Template]:
    """
    Build a template group; the last row (and every 4th from the end in
    larger groups) is held out.
    """
    out = []
    n = len(rows)
    for i, row in enumerate(rows):
        text, desc = row[0], row[1]
        tags_no = row[2] if len(row) > 2 else ()
        tags_need = row[3] if len(row) > 3 else ()
        heldout = (n - 1 - i) % 4 == 0
        out.append(
            _Template(
                f"{prefix}_{i}", src, dst, op, num, text, desc, tuple(tags_no), tuple(tags_need), heldout
            )
        )
    return out


_NM = ("measure",)
_START = []
_START += _group("start_count", "", "count", "", "start", [
    ("{H} has {n} {items}.", ""),
    ("{H} starts with {n} {items}.", ""),
    ("{H} collected {n} {items} last week.", ""),
    ("At first, {H} owns {n} {items}.", ""),
    ("{H} bought {n} {items} at the market.", ""),
    ("There are {n} {items} in {H}'s kitchen.", "", (), ("food",)),
    ("{H} keeps {n} {items} in a cupboard.", ""),
    ("{H} was given {n} {items} by {N2}.", ""),
])
_START += _group("start_groups", "", "groups", "", "small", [
    ("{H} buys {n} {containers} of {stuff}.", ""),
    ("{H} has {n} {containers} full of {stuff}.", ""),
    ("{H} orders {n} {containers} of {stuff} for a party.", ""),
    ("{H} brings home {n} {containers} of {stuff}.", ""),
    ("{H} receives {n} {containers} of {stuff}.", ""),
])
_START += _group("start_money", "", "money_have", "", "money", [
    ("{H} has ${n}.", ""),
    ("{H} saved ${n}.", ""),
    ("{H} has ${n} in {pos} wallet.", ""),
    ("{H}'s piggy bank holds ${n}.", ""),
    ("{H} starts the day with ${n}.", ""),
])
_START += _group("start_price", "", "price", "", "price", [
    ("{Items} cost ${n} each.", ""),
    ("One {item} costs ${n}.", ""),
    ("The shop sells {items} for ${n} apiece.", ""),
    ("Each {item} is priced at ${n}.", ""),
    ("A single {item} sells for ${n}.", ""),
])

_STEP = []
# count -> count
_STEP += _group("count_add", "count", "count", "+", "amount", [
    ("{H} buys {n} more {items}.", "{items} after buying more"),
    ("Then {H} finds {n} more {items}.", "{items} after finding more"),
    ("{N2} gives {obj} another {n} {items}.", "{items} after the gift"),
    ("Later that day, {H} gets {n} extra {items}.", "{items} now"),
    ("{H} adds {n} {items} to {pos} collection.", "{items} in total"),
    ("{H} receives {n} more {items} as a present.", "{items} after the present"),
    ("A neighbor drops off {n} more {items} for {obj}.", "{items} now"),
    ("{H} picks up {n} additional {items} on the way home.", "{items} in total"),
])
_STEP += _group("count_sub", "count", "count", "-", "sub", [
    ("{H} gives {n} {items} to {N2}.", "{items} left"),
    ("{H} sells {n} {items}.", "{items} left after selling"),
    ("{H} uses {n} {items} for a project.", "{items} remaining"),
    ("{n} of the {items} get lost.", "{items} left", _NM),
    ("{H} eats {n} of them.", "{items} left after eating", _NM, ("food",)),
    ("{H} donates {n} {items} to a school.", "{items} after donating"),
    ("{n} of the {items} break, so {H} throws them away.", "unbroken {items}", _NM, ("fragile",)),
    ("{N2} borrows {n} {items} from {obj}.", "{items} left"),
    ("{H} trades away {n} of the {items}.", "{items} remaining", _NM),
])
_STEP += _group("count_mul", "count", "count", "*", "mult", [
    ("By the end of the month, {H} has {n} times as many {items} as before.", "{items} at month end"),
    ("{H}'s supply of {items} then grows to {n} times its size.", "{items} after growing"),
    ("{H} ends up with {n} times the number of {items} {subj} had.", "{items} now"),
    ("A year later, {H} owns {n} times as many {items}.", "{items} a year later"),
])
_STEP += _group("count_div", "count", "count", "/", "div", [
    ("{H} splits the {items} into {n} equal piles and keeps one pile.", "{items} in one pile"),
    ("{H} shares the {items} equally among {n} people, {self} included.", "{items} per person"),
    ("{H} keeps only one of {n} equal portions of the {items}.", "{items} kept"),
    ("{H} divides the {items} evenly into {n} {containers} and takes one {container} home.", "{items} taken home"),
    ("{H} gives away all but one of {n} equal shares of the {items}.", "{items} in one share"),
])
# count <-> groups
_STEP += _group("count_pack", "count", "groups", "/", "div", [
    ("{H} packs the {items} into {containers} of {n}.", "{containers} filled"),
    ("Each {container} holds {n} {items}, and {H} fills {containers} with all of them.", "{containers} needed"),
    ("{H} puts {n} {items} in every {container}, using all of them.", "{containers} used"),
    ("{H} packs all the {items}, {n} per {container}.", "number of {containers}"),
])
_STEP += _group("groups_unpack", "groups", "count", "*", "small", [
    ("Each {container} contains {n} {items}.", "total {items}"),
    ("There are {n} {items} in every {container}.", "number of {items}"),
    ("Every {container} has {n} {items} inside.", "{items} in all"),
    ("One {container} holds {n} {items}.", "total {items}"),
])
_STEP += _group("groups_add", "groups", "groups", "+", "small", [
    ("{H} buys {n} more {containers}.", "{containers} now"),
    ("{N2} brings {obj} {n} extra {containers}.", "{containers} in total"),
    ("Later, {H} gets {n} more {containers} of {stuff}.", "{containers} after getting more"),
])
_STEP += _group("groups_sub", "groups", "groups", "-", "sub", [
    ("{H} sells {n} of the {containers}.", "{containers} kept"),
    ("{H} gives {n} {containers} to {N2}.", "{containers} left"),
    ("{n} of the {containers} are damaged and thrown out.", "good {containers}"),
    ("{H} returns {n} {containers} to the store.", "{containers} kept"),
])
# groups / count -> money
_STEP += _group("groups_cost", "groups", "money_spent", "*", "price", [
    ("Each {container} costs ${n}.", "cost of the {containers}"),
    ("{H} paid ${n} for every {container}.", "money spent on {containers}"),
    ("The {containers} were ${n} each.", "total cost"),
])
_STEP += _group("groups_sell", "groups", "money_earned", "*", "price", [
    ("{H} sells each {container} for ${n}.", "money from {containers}"),
    ("{H} sells the {containers} at ${n} apiece.", "sales"),
    ("Every {container} is sold for ${n}.", "money earned"),
])
_STEP += _group("count_sell", "count", "money_earned", "*", "price", [
    ("{H} sells all of the {items} for ${n} each.", "money from {items}"),
    ("Each {item} sells for ${n}, and {H} sells every one.", "money earned"),
    ("At ${n} per {item}, {H} sells them all.", "sales"),
    ("{H} sells the {items} at a price of ${n} each.", "earnings"),
])
_STEP += _group("count_cost", "count", "money_spent", "*", "price", [
    ("{H} paid ${n} for each {item}.", "cost of the {items}"),
    ("Each {item} cost {obj} ${n}.", "money spent"),
    ("The {items} cost ${n} apiece.", "total cost"),
])
# money -> money
_STEP += _group("earn_add", "money_earned", "money_earned", "+", "money", [
    ("{H} also gets a ${n} tip.", "earnings with tip"),
    ("{H} earns another ${n} from {job}.", "total earnings"),
    ("On top of that, {H} makes ${n} from {job}.", "money earned in all"),
])
_STEP += _group("earn_mul", "money_earned", "money_earned", "*", "mult", [
    ("{H} does this every week for {n} weeks.", "earnings over the weeks"),
    ("{H} repeats the same sale {n} times.", "total earnings"),
    ("This happens on {n} days in a row.", "money earned in all"),
])
_STEP += _group("earn_spend", "money_earned", "money_have", "-", "sub", [
    ("{H} then spends ${n} on {thing}.", "money left"),
    ("Out of that money, {H} pays ${n} for {thing}.", "money remaining"),
    ("{H} uses ${n} of it to buy {thing}.", "money left over"),
])
_STEP += _group("earn_share", "money_earned", "money_share", "/", "div", [
    ("{H} shares the money equally among {n} friends.", "money per friend"),
    ("The money is split evenly between {n} team members.", "share per member"),
    ("{H} divides the money equally into {n} envelopes.", "money per envelope"),
])
_STEP += _group("spent_add", "money_spent", "money_spent", "+", "money", [
    ("{H} also pays ${n} for {thing}.", "total spent"),
    ("There is an extra ${n} delivery fee.", "cost with delivery"),
    ("{H} also buys {thing} for ${n}.", "money spent in all"),
])
_STEP += _group("spent_sub", "money_spent", "money_spent", "-", "sub", [
    ("{H} gets a ${n} discount.", "cost after discount"),
    ("A coupon takes ${n} off the bill.", "cost after coupon"),
    ("The store refunds {obj} ${n}.", "net cost"),
])
_STEP += _group("spent_mul", "money_spent", "money_spent", "*", "mult", [
    ("{H} does the same shopping every week for {n} weeks.", "spent over the weeks"),
    ("{H} makes the same purchase {n} times.", "total spent"),
    ("This purchase is repeated on {n} days.", "money spent in all"),
])
_STEP += _group("spent_share", "money_spent", "money_share", "/", "div", [
    ("{H} splits the cost equally with friends, {n} people in all.", "cost per person"),
    ("The bill is shared evenly by {n} people.", "share of the bill"),
    ("{H} pays it off in {n} equal installments.", "installment"),
])
_STEP += _group("have_sub", "money_have", "money_have", "-", "sub", [
    ("{H} spends ${n} on {thing}.", "money left"),
    ("{H} pays ${n} for {thing}.", "money remaining"),
    ("{H} buys {thing} for ${n}.", "money left over"),
    ("{H} lends ${n} to {N2}.", "money left"),
])
_STEP += _group("have_add", "money_have", "money_have", "+", "money", [
    ("{H} earns ${n} from {job}.", "money now"),
    ("{N2} gives {obj} ${n}.", "money after the gift"),
    ("{H} finds ${n} in an old coat.", "money in total"),
    ("{H} gets ${n} for {pos} birthday.", "money now"),
])
_STEP += _group("have_mul", "money_have", "money_have", "*", "mult", [
    ("After a year of saving, {H} has {n} times as much money.", "money after saving"),
    ("{H}'s savings grow to {n} times the amount.", "savings now"),
    ("{H} manages to multiply the money by {n}.", "money now"),
])
_STEP += _group("have_buy", "money_have", "count", "/", "price_div", [
    ("{H} spends all of it on {items} that cost ${n} each.", "{items} bought"),
    ("With all that money, {H} buys {items} at ${n} apiece.", "number of {items}"),
    ("{H} uses all the money to buy {items} for ${n} each.", "{items} bought"),
])
_STEP += _group("have_share", "money_have", "money_share", "/", "div", [
    ("{H} splits the money equally among {n} children.", "money per child"),
    ("{H} divides the money evenly into {n} jars.", "money per jar"),
    ("{H} spends the money evenly over {n} days.", "money per day"),
])
# price -> price
_STEP += _group("price_add", "price", "price", "+", "money", [
    ("Then the price goes up by ${n}.", "new price"),
    ("The next month, each {item} costs ${n} more.", "price after the increase"),
    ("A tax adds ${n} to the price of each {item}.", "price with tax"),
])
_STEP += _group("price_sub", "price", "price", "-", "sub", [
    ("During a sale, the price drops by ${n}.", "sale price"),
    ("There is a ${n} discount on each {item}.", "discounted price"),
    ("Members pay ${n} less per {item}.", "member price"),
])
_STEP += _group("price_mul", "price", "price", "*", "mult", [
    ("A year later, the price is {n} times higher.", "new price"),
    ("At the airport, {items} cost {n} times as much.", "airport price"),
    ("The fancy version costs {n} times the regular price.", "fancy price"),
])

# Join nodes (two quantities): (left sem, right sem) -> dst.
_JOIN = []
_JOIN += _group("join_total", "count+count", "count_both", "+", "", [
    ("", "{items} together"), ("", "total {items}"), ("", "{items} in all"),
])
_JOIN += _group("join_diff", "count-count", "count_diff", "-", "", [
    ("", "difference"), ("", "how many more"), ("", "extra {items}"),
])
_JOIN += _group("join_cost", "count*price", "money_total", "*", "", [
    ("", "total cost"), ("", "cost of all {items}"), ("", "money needed"),
])

# Questions, keyed by the semantic type of the final quantity.
_QUESTIONS: Dict[str, List[_Template]] = {}


def _qgroup(sem: str, texts: Sequence[str]) -> None:
    _QUESTIONS[sem] = _group(f"q_{sem}", sem, sem, "", "", [(t, "") for t in texts])


_qgroup("count", [
    "How many {items} does {H} have now?",
    "How many {items} does {H} end up with?",
    "How many {items} does {H} have in the end?",
    "What is the number of {items} {H} has now?",
    "How many {items} does {H} still have?",
])
_qgroup("groups", [
    "How many {containers} does {H} have?",
    "How many {containers} are there now?",
    "What is the number of {containers}?",
    "How many {containers} does {H} end up with?",
])
_qgroup("money_earned", [
    "How much money does {H} make?",
    "How many dollars does {H} earn?",
    "How much does {H} earn in total?",
    "What are {H}'s total earnings in dollars?",
])
_qgroup("money_spent", [
    "How much does {H} spend?",
    "How much money does {H} pay in total?",
    "What is the total cost in dollars?",
    "How many dollars does {H} spend altogether?",
])
_qgroup("money_have", [
    "How much money does {H} have now?",
    "How many dollars does {H} have left?",
    "How much money does {H} have in the end?",
    "How much money does {H} end up with?",
])
_qgroup("money_share", [
    "How much is each share in dollars?",
    "How many dollars is each part?",
    "How much money is in each part?",
])
_qgroup("price", [
    "What is the price of one {item} now?",
    "How much does one {item} cost now?",
    "What does a single {item} cost in dollars?",
])
_qgroup("count_both", [
    "How many {items} do {H} and {H2} have together?",
    "How many {items} do they have altogether?",
    "What is the total number of {items} {H} and {H2} have?",
    "How many {items} do {H} and {H2} have combined?",
])
_qgroup("count_diff", [
    "How many more {items} does {Hbig} have than {Hsmall}?",
    "How many fewer {items} does {Hsmall} have than {Hbig}?",
    "By how many {items} does {Hbig} have more than {Hsmall}?",
])
_qgroup("money_total", [
    "How much would all of {H}'s {items} cost at that price?",
    "What is the cost in dollars of all of {H}'s {items} at the final price?",
    "How much money would it take to buy that many {items} at that price?",
])

_DISTRACTORS = _group("distractor", "", "", "", "", [
    ("{N3} has {d} {items2}.", ""),
    ("{H}'s {relative} is {d} years old.", ""),
    ("{place_cap} is {d} miles away from {H}'s home.", ""),
    ("{N3} owns {d} {pets}.", ""),
    ("Last year, {H} had {d} {items2}.", ""),
    ("The store has {d} employees.", ""),
    ("{N3} spent ${d} on {thing} yesterday.", ""),
    ("It takes {H} {d} minutes to walk to {place}.", ""),
    ("{H}'s {relative} has {d} {items2}.", ""),
    ("There are {d} people waiting in line.", ""),
])

_ALL_TEMPLATES = _START + _STEP + _JOIN + [t for g in _QUESTIONS.values() for t in g] + _DISTRACTORS
assert len({t.tid for t in _ALL_TEMPLATES}) == len(_ALL_TEMPLATES)


def template_ids(split: str = "all") -> List[str]:
    """
    Return the ids of all templates usable with `split`.
    """
    if split not in TEMPLATE_SPLITS:
        raise ValueError(f"Unknown template split '{split}'")
    return [t.tid for t in _ALL_TEMPLATES if _usable(t, split)]


def _usable(t: _Template, split: str) -> bool:
    if split == "all":
        return True
    return t.heldout == (split == "heldout")


def max_distractors(split: str = "train") -> int:
    """
    Return the number of distinct distractor templates usable with `split`.
    """
    return sum(_usable(t, split) for t in _DISTRACTORS)


# #############################################################################
# Generation
# #############################################################################

_START_GROUP = {"count": "start_count", "groups": "start_groups", "money_have": "start_money", "price": "start_price"}


@dataclasses.dataclass
class _Ctx:
    """
    Rendering context of one problem.
    """

    rng: random.Random
    split: str
    item: _Item
    max_number: int
    max_value: int
    names: List[Tuple[str, str]]
    use_pronouns: bool = True
    graph: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    steps: List[str] = dataclasses.field(default_factory=list)
    sentences: List[str] = dataclasses.field(default_factory=list)
    tids: List[str] = dataclasses.field(default_factory=list)
    numbers: List[int] = dataclasses.field(default_factory=list)


def _pick(rng: random.Random, seq: Sequence) -> Any:
    return seq[rng.randrange(len(seq))]


def _ok_for_item(t: _Template, item: _Item) -> bool:
    if any(tag in item.tags for tag in t.tags_no):
        return False
    return all(tag in item.tags for tag in t.tags_need)


def _fill(
    ctx: _Ctx,
    text: str,
    holder: Tuple[str, str],
    n: Optional[int],
    pronoun: bool = False,
    **extra: Any,
) -> str:
    """
    Fill a template's slots and capitalize the sentence.

    :param pronoun: refer to the holder with a pronoun instead of the name
    """
    name, gender = holder
    pr = _PRONOUNS[gender]
    if pronoun:
        text = text.replace("{H}'s", "{pos}").replace("{H}", "{subj}")
    slots = {
        "H": name,
        "n": "" if n is None else str(n),
        "items": ctx.item.plural,
        "Items": ctx.item.plural[0].upper() + ctx.item.plural[1:],
        "item": ctx.item.singular,
        "container": ctx.item.container,
        "containers": ctx.item.containers,
        "stuff": ctx.item.plural.rsplit(" of ", 1)[-1],
        # names[0] and names[1] are protagonists; names[2] is a bystander.
        "N2": ctx.names[2][0],
        "thing": _pick(ctx.rng, _THINGS),
        "job": _pick(ctx.rng, _JOBS),
        **pr,
    }
    slots.update(extra)
    s = text.format(**slots)
    return s[0].upper() + s[1:]


def _sample_number(ctx: _Ctx, t: _Template, value: int) -> Optional[int]:
    """
    Sample the number given in a node sentence so the result stays a
    non-negative integer <= max_value. Returns None if impossible.
    """
    rng, mx, mv = ctx.rng, ctx.max_number, ctx.max_value
    kind = t.num
    if kind == "start":
        return rng.randint(5, max(5, mx))
    if kind == "money":
        return rng.randint(5, max(5, mx)) if t.op == "" else rng.randint(1, max(1, min(mx, mv - value)))
    if kind == "amount":
        hi = min(mx, mv - value)
        return rng.randint(1, hi) if hi >= 1 else None
    if kind == "small":
        hi = min(20, mx)
        if t.op == "+":
            hi = min(hi, mv - value)
            return rng.randint(1, hi) if hi >= 1 else None
        if t.op == "*":
            hi = min(hi, mv // max(value, 1))
            return rng.randint(2, hi) if hi >= 2 else None
        return rng.randint(2, hi)
    if kind == "price":
        lo, hi = ctx.item.price
        if t.op == "":
            return rng.randint(lo, hi)
        if t.src == "groups":
            lo, hi = lo * 5, hi * 12
        hi = min(hi, mv // max(value, 1))
        return rng.randint(lo, hi) if hi >= lo else None
    if kind == "mult":
        hi = min(10, mv // max(value, 1))
        return rng.randint(2, hi) if hi >= 2 else None
    if kind == "sub":
        hi = min(mx, value - 1)
        return rng.randint(1, hi) if hi >= 1 else None
    if kind in ("div", "price_div"):
        lo = 2 if kind == "div" else ctx.item.price[0]
        divs = [d for d in range(max(2, lo), min(value, 25) + 1) if value % d == 0 and value // d >= 1]
        if kind == "div":
            divs = [d for d in divs if d < value]
        return _pick(rng, divs) if divs else None
    raise ValueError(kind)


def _apply(op: str, a: int, b: int) -> int:
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    assert a % b == 0
    return a // b


def _start(ctx: _Ctx, sem: str, holder: Tuple[str, str]) -> Tuple[str, int]:
    """
    Add a start sentence for a quantity of type `sem`.

    :return: (node reference, value)
    """
    group = _START_GROUP[sem]
    cands = [
        t for t in _START
        if t.tid.rsplit("_", 1)[0] == group and _usable(t, ctx.split) and _ok_for_item(t, ctx.item)
    ]
    t = _pick(ctx.rng, cands)
    if sem == "count":
        n = ctx.rng.randint(5, max(5, ctx.max_number))
    elif sem == "groups":
        n = ctx.rng.randint(2, min(20, max(2, ctx.max_number)))
    elif sem == "money_have":
        n = ctx.rng.randint(10, max(10, ctx.max_number * 5))
    else:
        n = _sample_number(ctx, t, 0)
    ctx.sentences.append(_fill(ctx, t.text, holder, n))
    ctx.tids.append(t.tid)
    ctx.numbers.append(n)
    return f"g{len(ctx.numbers) - 1}", n


def _chain(
    ctx: _Ctx,
    sem: str,
    ref: str,
    value: int,
    length: int,
    holder: Tuple[str, str],
    allowed_dst: Optional[Sequence[str]] = None,
    final_dst: Optional[Sequence[str]] = None,
) -> Optional[Tuple[str, str, int]]:
    """
    Extend a quantity by `length` nodes.

    :param allowed_dst: semantic types every node may produce (None: any
        non-terminal type)
    :param final_dst: semantic types the last node may produce (None: any)
    :return: (sem, node reference, value) of the last quantity, or None
    """
    rng = ctx.rng
    for i in range(length):
        last = i == length - 1
        cands = []
        for t in _STEP:
            if t.src != sem or not _usable(t, ctx.split) or not _ok_for_item(t, ctx.item):
                continue
            if allowed_dst is not None and t.dst not in allowed_dst:
                continue
            if not last and t.dst == "money_share":
                continue
            if last and final_dst is not None and t.dst not in final_dst:
                continue
            cands.append(t)
        if not cands:
            return None
        rng.shuffle(cands)
        item_before = ctx.item
        for t in cands:
            if t.tid.startswith("have_buy"):
                # Money is spent on a new kind of item.
                others = [it for it in _ITEMS if it != item_before and "measure" not in it.tags]
                ctx.item = _pick(rng, others)
            n = _sample_number(ctx, t, value)
            if n is not None:
                break
            ctx.item = item_before
        else:
            return None
        new = _apply(t.op, value, n)
        if new < 0 or new > ctx.max_value:
            return None
        pronoun = ctx.use_pronouns and bool(ctx.sentences) and rng.random() < 0.4
        ctx.sentences.append(_fill(ctx, t.text, holder, n, pronoun=pronoun))
        ctx.tids.append(t.tid)
        ctx.numbers.append(n)
        desc = _fill(ctx, t.desc, holder, n).lower()
        node = f"s{len(ctx.graph)}"
        ctx.graph.append({"id": node, "op": t.op, "args": [ref, n], "value": new, "tid": t.tid})
        ctx.steps.append(f"{desc}: {value} {t.op} {n} = {new}")
        sem, ref, value = t.dst, node, new
    return sem, ref, value


def _question(ctx: _Ctx, sem: str, **slots: Any) -> None:
    cands = [t for t in _QUESTIONS[sem] if _usable(t, ctx.split)]
    t = _pick(ctx.rng, cands)
    holder = slots.pop("holder")
    ctx.sentences.append(_fill(ctx, t.text, holder, None, **slots))
    ctx.tids.append(t.tid)


def _join_node(ctx: _Ctx, kind: str, left: Tuple[str, int], right: Tuple[str, int]) -> int:
    t = _pick(ctx.rng, [t for t in _JOIN if t.tid.startswith(kind) and _usable(t, ctx.split)])
    (lref, lv), (rref, rv) = left, right
    new = _apply(t.op, lv, rv)
    node = f"s{len(ctx.graph)}"
    ctx.graph.append({"id": node, "op": t.op, "args": [lref, rref], "value": new, "tid": t.tid})
    desc = t.desc.format(items=ctx.item.plural)
    ctx.steps.append(f"{desc}: {lv} {t.op} {rv} = {new}")
    ctx.tids.append(t.tid)
    return new


def _try_generate(
    ctx: _Ctx, n_steps: int, join_prob: float
) -> Optional[int]:
    """
    Build the graph and sentences (without distractors and question order).

    :return: the answer, or None on failure
    """
    rng = ctx.rng
    h1, h2 = ctx.names[0], ctx.names[1]
    if rng.random() < join_prob:
        ctx.use_pronouns = False
        n1 = rng.randint(0, n_steps - 1)
        n2 = n_steps - 1 - n1
        if rng.random() < 0.6:
            # Two people with counts of the same item.
            ref1, v1 = _start(ctx, "count", h1)
            res1 = _chain(ctx, "count", ref1, v1, n1, h1, allowed_dst=("count",))
            if res1 is None:
                return None
            ref2, v2 = _start(ctx, "count", h2)
            res2 = _chain(ctx, "count", ref2, v2, n2, h2, allowed_dst=("count",))
            if res2 is None:
                return None
            (_, r1, v1), (_, r2, v2) = res1, res2
            if rng.random() < 0.5 or v1 == v2:
                ans = _join_node(ctx, "join_total", (r1, v1), (r2, v2))
                if ans > ctx.max_value:
                    return None
                _question(ctx, "count_both", holder=h1, H2=h2[0])
            else:
                big, small = ((r1, v1), (r2, v2)) if v1 > v2 else ((r2, v2), (r1, v1))
                hb, hs = (h1, h2) if v1 > v2 else (h2, h1)
                ans = _join_node(ctx, "join_diff", big, small)
                _question(ctx, "count_diff", holder=h1, Hbig=hb[0], Hsmall=hs[0])
            return ans
        # A count and a unit price.
        start_sem = rng.choice(("count", "groups"))
        ref1, v1 = _start(ctx, start_sem, h1)
        if start_sem == "groups" and n1 == 0:
            return None
        res1 = _chain(ctx, start_sem, ref1, v1, n1, h1, allowed_dst=("count", "groups"), final_dst=("count",))
        if res1 is None or res1[0] != "count":
            return None
        refp, vp = _start(ctx, "price", h1)
        resp = _chain(ctx, "price", refp, vp, n2, h1, allowed_dst=("price",))
        if resp is None or resp[2] < 1:
            return None
        ans = _join_node(ctx, "join_cost", (res1[1], res1[2]), (resp[1], resp[2]))
        if ans > ctx.max_value:
            return None
        _question(ctx, "money_total", holder=h1)
        return ans
    start_sem = rng.choice(("count", "count", "groups", "money_have"))
    ref, v = _start(ctx, start_sem, h1)
    res = _chain(ctx, start_sem, ref, v, n_steps, h1)
    if res is None:
        return None
    sem, _, ans = res
    _question(ctx, sem, holder=h1)
    return ans


def _add_distractors(ctx: _Ctx, k: int, holder: Tuple[str, str]) -> None:
    """
    Insert `k` irrelevant sentences before the question.
    """
    rng = ctx.rng
    used = set()
    for _ in range(k):
        cands = [t for t in _DISTRACTORS if t.tid not in used and _usable(t, ctx.split)]
        t = _pick(rng, cands)
        used.add(t.tid)
        others = [it for it in _ITEMS if it.plural != ctx.item.plural]
        place = _pick(rng, _PLACES)
        d = rng.randint(2, 60)
        text = _fill(
            ctx,
            t.text,
            holder,
            None,
            d=d,
            N3=ctx.names[3][0],
            items2=_pick(rng, others).plural,
            relative=_pick(rng, _RELATIVES),
            place=place,
            place_cap=place[0].upper() + place[1:],
            pets=_pick(rng, _PETS),
        )
        pos = rng.randint(0, len(ctx.sentences))
        ctx.sentences.insert(pos, text)
        ctx.tids.append(t.tid)


def generate_example(
    rng: random.Random,
    n_steps: int,
    *,
    n_distractors: int = 0,
    template_split: str = "train",
    max_number: int = 100,
    max_value: int = 10_000,
    join_prob: float = 0.3,
) -> lsform.Example:
    """
    Generate one M3 word problem with exactly `n_steps` solution steps.

    :param rng: random generator (controls reproducibility)
    :param n_steps: number of graph nodes = solution steps (>= 1)
    :param n_distractors: number of irrelevant sentences to insert
    :param template_split: "train", "heldout" or "all" templates
    :param max_number: bound on most numbers given in sentences
    :param max_value: bound on every intermediate value
    :param join_prob: probability of a two-chain graph joined at the end
    :return: math example with one described step per graph node
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    if template_split not in TEMPLATE_SPLITS:
        raise ValueError(f"Unknown template split '{template_split}'")
    if not 0 <= n_distractors <= max_distractors(template_split):
        raise ValueError(f"n_distractors must be in [0, {max_distractors(template_split)}]")
    if max_number < 5 or max_value < max_number:
        raise ValueError("Need max_number >= 5 and max_value >= max_number")
    for _ in range(10_000):
        genders = [rng.choice("fm") for _ in range(3)]
        # The distractor bystander has the other gender, so pronouns after
        # a distractor sentence still refer to the protagonist.
        genders.append("m" if genders[0] == "f" else "f")
        names: List[Tuple[str, str]] = []
        for g in genders:
            pool = _NAMES_F if g == "f" else _NAMES_M
            nm = _pick(rng, pool)
            while nm in [x for x, _ in names]:
                nm = _pick(rng, pool)
            names.append((nm, g))
        ctx = _Ctx(
            rng=rng,
            split=template_split,
            item=_pick(rng, _ITEMS),
            max_number=max_number,
            max_value=max_value,
            names=names,
        )
        first_item = ctx.item
        ans = _try_generate(ctx, n_steps, join_prob)
        if ans is None or len(ctx.steps) != n_steps:
            continue
        question = ctx.sentences.pop()
        _add_distractors(ctx, n_distractors, names[0])
        problem = " ".join(ctx.sentences + [question])
        return lsform.Example(
            task="math",
            problem=problem,
            steps=list(ctx.steps),
            answer=str(ans),
            meta={
                "level": "M3",
                "n_steps": n_steps,
                "n_distractors": n_distractors,
                "template_split": template_split,
                "template_ids": list(ctx.tids),
                "graph": ctx.graph,
                "given": list(ctx.numbers),
                "shape": "join" if ctx.graph[-1]["tid"].startswith("join") else "chain",
                "names": [nm for nm, _ in names],
                "item": first_item.plural,
            },
        )
    raise RuntimeError(f"Could not generate an M3 example with n_steps={n_steps}")


def evaluate_graph(graph: List[Dict[str, Any]], given: Sequence[int]) -> int:
    """
    Re-evaluate a graph from `meta["graph"]` and `meta["given"]`.

    Arguments are node ids `s<i>`, given-number ids `g<i>` (index into
    `given`), or integer literals.

    :return: the value of the last node
    """
    vals: Dict[str, int] = {}
    for node in graph:
        args = []
        for a in node["args"]:
            if isinstance(a, str) and a.startswith("s"):
                args.append(vals[a])
            elif isinstance(a, str):
                args.append(given[int(a[1:])])
            else:
                args.append(a)
        vals[node["id"]] = _apply(node["op"], *args)
    return vals[graph[-1]["id"]]


def generate_dataset(
    n: int,
    seed: int,
    *,
    n_steps_range: Tuple[int, int] = (1, 4),
    n_distractors_range: Tuple[int, int] = (0, 0),
    template_split: str = "train",
    max_number: int = 100,
    max_value: int = 10_000,
    join_prob: float = 0.3,
    exclude: Optional[set] = None,
) -> Iterator[lsform.Example]:
    """
    Yield `n` unique M3 examples with `n_steps` uniform in `n_steps_range`
    and the number of distractors uniform in `n_distractors_range`.

    :param exclude: problem strings to skip (e.g. a held-out test set)
    """
    rng = random.Random(seed)
    seen = set(exclude or ())
    made = 0
    attempts = 0
    while made < n:
        attempts += 1
        if attempts > 50 * n + 1000:
            raise RuntimeError(f"Only {made}/{n} unique examples; widen the ranges")
        ex = generate_example(
            rng,
            rng.randint(*n_steps_range),
            n_distractors=rng.randint(*n_distractors_range),
            template_split=template_split,
            max_number=max_number,
            max_value=max_value,
            join_prob=join_prob,
        )
        if ex.problem in seen:
            continue
        seen.add(ex.problem)
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


problem_from_record = lsmam1.problem_from_record


def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-steps", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=4)
    ap.add_argument("--min-distractors", type=int, default=0)
    ap.add_argument("--max-distractors", type=int, default=0)
    ap.add_argument("--template-split", default="train", choices=TEMPLATE_SPLITS)
    ap.add_argument("--max-number", type=int, default=100)
    ap.add_argument("--max-value", type=int, default=10_000)
    ap.add_argument("--join-prob", type=float, default=0.3)
    ap.add_argument("--out", required=True, help="output .jsonl path")
    ap.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="JSONL files whose problems must not be generated (test sets)",
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
            n_steps_range=(a.min_steps, a.max_steps),
            n_distractors_range=(a.min_distractors, a.max_distractors),
            template_split=a.template_split,
            max_number=a.max_number,
            max_value=a.max_value,
            join_prob=a.join_prob,
            exclude=exclude,
        ):
            f.write(json.dumps(example_to_record(ex)) + "\n")


if __name__ == "__main__":
    _main()
