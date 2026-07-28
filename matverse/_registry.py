"""The function registry — matverse's public API described to a machine.

Every public function in matverse carries a ``@register_function`` entry giving
its aliases, a description, the state it consumes and the state it creates. The
registry indexes those entries so an agent can ask "how do I compute
thermodynamic stability" and get back the function, its signature, a runnable
example, and what it will write into the object.

Why this can exist here
-----------------------
The two contract slots that matter most, ``requires`` and ``produces``, name
state that a call consumes and creates. They only bind in a library where calls
*have* named state to point at. ``pymatgen`` adds and removes none — results are
attributes on returned objects — so on ``pymatgen-analysis-defects`` those slots
had to be replaced with an object-centric substitute. matverse's rule that
operations deposit into named containers is what gives them referents, and that
is deliberate rather than incidental.

Templated slot names
--------------------
Contract slots hold the **literal template**, not the resolved key::

    produces={"obs": ["e_above_hull_{level}"]}

not ``obs['e_above_hull_emt']``. The template is what holds for every call; a
resolved key is true only for the one call that produced it, and the pattern is
the part worth teaching. ``matverse._probe`` resolves templates against a call's
bound arguments before checking them, so the two agree by construction.

This module is vendored on purpose: matverse depends on anndata, mudata, numpy,
pandas, pymatgen and ase, and on nothing else in its core path.
"""

from __future__ import annotations

import inspect
import re
from difflib import get_close_matches
from functools import wraps
from typing import Any, Callable, Dict, Iterable, List, Optional

# Containers a matverse contract may name. AnnData's own axes, plus the four
# ``uns`` sub-containers the slot convention treats as first-class.
CONTRACT_KEYS: frozenset[str] = frozenset({
    "obs", "var", "obsm", "varm", "obsp", "varp", "layers", "X", "uns",
    "mod",          # MuData modality
    "structures",   # uns['structures'][variant]
    "levels",       # uns['levels'][level]
    "features",     # uns['features'][block]
    "files",        # written to disk rather than into the object
})

_TEMPLATE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def slot_template_fields(slot: str) -> List[str]:
    """Parameter names a slot template interpolates, e.g. ``['level']``."""
    return _TEMPLATE.findall(slot)


def resolve_slot(slot: str, bound: Dict[str, Any]) -> str:
    """Resolve ``'e_above_hull_{level}'`` against bound call arguments.

    Unknown fields are left as-is rather than raising, so a partially
    resolvable template still yields something a probe can report on.
    """
    out = slot
    for field in slot_template_fields(slot):
        if field in bound and bound[field] is not None:
            out = out.replace("{" + field + "}", str(bound[field]))
    return out


class AliasCollision(ValueError):
    """Raised when two functions claim the same alias.

    Machine-generated registries collide constantly — 46 of 1,162 aliases did on
    omicverse — and a silent overwrite means one function becomes unreachable
    through the channel the registry exists to provide. Failing at import is
    noisy and cheap; failing at retrieval time is quiet and expensive.
    """


class FunctionRegistry:
    """An in-memory index of decorated functions, searchable by intent."""

    def __init__(self) -> None:
        self._entries: Dict[str, Dict[str, Any]] = {}   # full_name -> entry
        self._index: Dict[str, str] = {}                # alias/short -> full_name
        self._categories: Dict[str, List[str]] = {}

    # -- registration --------------------------------------------------

    def register(self, func: Callable, *, aliases: List[str], category: str,
                 description: str, examples: Optional[List[str]] = None,
                 related: Optional[List[str]] = None,
                 requires: Optional[Dict[str, List[str]]] = None,
                 produces: Optional[Dict[str, List[str]]] = None,
                 prerequisites: Optional[List[str]] = None,
                 dispatch: Optional[str] = None,
                 notes: str = "") -> Callable:
        if not aliases or not all(str(a).strip() for a in aliases):
            raise ValueError("registration requires at least one non-empty alias")
        if not category.strip():
            raise ValueError("registration requires a category")
        if not description.strip():
            raise ValueError("registration requires a description")

        for kind, mapping in (("requires", requires), ("produces", produces)):
            if mapping is None:
                continue
            if not isinstance(mapping, dict):
                raise TypeError(f"{kind} must be a dict of container -> [slot, ...]")
            bad = set(mapping) - CONTRACT_KEYS
            if bad:
                raise ValueError(
                    f"{kind} names unknown container(s) {sorted(bad)}; "
                    f"allowed: {sorted(CONTRACT_KEYS)}")

        full_name = f"{func.__module__}.{func.__qualname__}"
        short_name = func.__name__
        try:
            signature = str(inspect.signature(func))
        except (TypeError, ValueError):
            signature = "(...)"

        public_name = _public_name(func)
        keys = [a.strip().lower() for a in aliases] + [short_name.lower(),
                                                       full_name.lower(),
                                                       public_name.lower()]
        for key in keys:
            owner = self._index.get(key)
            if owner is not None and owner != full_name:
                raise AliasCollision(
                    f"alias {key!r} is already registered to {owner}; "
                    f"{full_name} cannot also claim it")

        entry = {
            "function": func,
            "full_name": full_name,
            "short_name": short_name,
            "public_name": public_name,
            "module": func.__module__,
            "aliases": [a.strip() for a in aliases],
            "category": category,
            "description": description.strip(),
            "examples": list(examples or []),
            "related": list(related or []),
            "requires": dict(requires or {}),
            "produces": dict(produces or {}),
            "prerequisites": list(prerequisites or []),
            "dispatch": dispatch,
            "notes": notes,
            "signature": signature,
            "docstring": inspect.getdoc(func) or "",
        }
        self._entries[full_name] = entry
        for key in keys:
            self._index[key] = full_name
        self._categories.setdefault(category, [])
        if full_name not in self._categories[category]:
            self._categories[category].append(full_name)
        return func

    # -- lookup --------------------------------------------------------

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        full = self._index.get(name.strip().lower())
        return self._entries.get(full) if full else None

    def find(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Rank entries against a free-text intent."""
        q = (query or "").strip().lower()
        if not q:
            return []
        exact = self.get(q)
        scored: List[tuple[float, Dict[str, Any]]] = []
        tokens = _tokens(q)
        for entry in self._entries.values():
            if exact is not None and entry is exact:
                continue
            score = 0.0
            for alias in entry["aliases"]:
                a = alias.lower()
                if a == q:
                    score += 10.0
                elif q in a or a in q:
                    score += 4.0
                score += 2.0 * _overlap(tokens, _tokens(a))
            score += 3.0 * _overlap(tokens, _tokens(entry["short_name"]))
            score += 1.0 * _overlap(tokens, _tokens(entry["description"]))
            score += 0.5 * _overlap(tokens, _tokens(entry["category"]))
            for slot in _all_slots(entry["produces"]):
                score += 1.5 * _overlap(tokens, _tokens(slot))
            if score:
                scored.append((score, entry))
        scored.sort(key=lambda pair: (-pair[0], pair[1]["full_name"]))
        out = ([exact] if exact is not None else []) + [e for _, e in scored]
        if not out:
            near = get_close_matches(q, list(self._index), n=limit, cutoff=0.6)
            out = [self._entries[self._index[n]] for n in near]
        return out[:limit]

    def producers_of(self, slot: str) -> List[Dict[str, Any]]:
        """Which functions write a slot. Answers 'where does this come from?'.

        Matches a concrete key against templated claims, so asking for
        ``e_above_hull_emt`` finds the function declaring
        ``e_above_hull_{level}``.
        """
        want = slot.strip()
        hits = []
        for entry in self._entries.values():
            for candidate in _all_slots(entry["produces"]):
                if candidate == want or _slot_matches(candidate, want):
                    hits.append(entry)
                    break
        return hits

    def categories(self) -> Dict[str, List[str]]:
        return {k: list(v) for k, v in self._categories.items()}

    def entries(self) -> List[Dict[str, Any]]:
        return list(self._entries.values())

    def __len__(self) -> int:
        return len(self._entries)

    # -- rendering -----------------------------------------------------

    def describe(self, name: str) -> str:
        """The block an agent is handed when it looks a function up."""
        entry = self.get(name)
        if entry is None:
            near = ", ".join(e["public_name"] for e in self.find(name, limit=3))
            return f"no entry for {name!r}" + (f"; did you mean: {near}" if near else "")
        lines = [f"{entry['public_name']}{entry['signature']}", "",
                 entry["description"]]
        if entry["dispatch"]:
            lines += ["", f"dispatch: {entry['dispatch']}"]
        for label, mapping in (("requires", entry["requires"]),
                               ("produces", entry["produces"])):
            if mapping:
                lines.append("")
                lines.append(f"{label}:")
                for container, slots in mapping.items():
                    for slot in slots:
                        lines.append(f"  {_render_slot(container, slot)}")
        if entry["prerequisites"]:
            lines += ["", "run first: " + ", ".join(entry["prerequisites"])]
        if entry["examples"]:
            lines += ["", "examples:"] + [f"  {ex}" for ex in entry["examples"]]
        if entry["related"]:
            lines += ["", "related: " + ", ".join(entry["related"])]
        if entry["notes"]:
            lines += ["", entry["notes"]]
        return "\n".join(lines)


_REGISTRY = FunctionRegistry()


def get_registry() -> FunctionRegistry:
    return _REGISTRY


def register_function(*, aliases: List[str], category: str, description: str,
                      examples: Optional[List[str]] = None,
                      related: Optional[List[str]] = None,
                      requires: Optional[Dict[str, List[str]]] = None,
                      produces: Optional[Dict[str, List[str]]] = None,
                      prerequisites: Optional[List[str]] = None,
                      dispatch: Optional[str] = None,
                      notes: str = "") -> Callable:
    """Promote a function to a registry entry.

    ``requires`` and ``produces`` map a container to the slots the call reads
    and writes, and may interpolate the call's own parameters::

        @register_function(
            aliases=["convex hull", "energy above hull"],
            category="thermo",
            description="Distance above the convex hull at one level of theory.",
            requires={"obs": ["energy_{level}"], "structures": ["{source}"]},
            produces={"obs": ["e_above_hull_{level}"], "levels": ["{level}"]},
            prerequisites=["mv.calc.energy"],
            examples=["mv.thermo.hull(md, level='emt')"],
        )

    Every claim made here is checked by execution in ``matverse._probe``; claims
    that fail their probe are deleted rather than repaired by hand.
    """
    def decorator(func: Callable) -> Callable:
        _REGISTRY.register(func, aliases=aliases, category=category,
                           description=description, examples=examples,
                           related=related, requires=requires, produces=produces,
                           prerequisites=prerequisites, dispatch=dispatch,
                           notes=notes)

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        wrapper._registry_entry = _REGISTRY.get(f"{func.__module__}.{func.__qualname__}")
        return wrapper
    return decorator


# -- helpers -----------------------------------------------------------

def _public_name(func: Callable) -> str:
    """``matverse.thermo.hull`` -> ``mv.thermo.hull``."""
    module = func.__module__
    if module.startswith("matverse."):
        return "mv." + module[len("matverse."):] + "." + func.__name__
    return f"{module}.{func.__name__}"


def _render_slot(container: str, slot: str) -> str:
    if container == "structures":
        return f"obsm['structures'][{slot!r}]"
    if container in ("levels", "features"):
        return f"uns['{container}'][{slot!r}]"
    if container == "X":
        return "X"
    return f"{container}[{slot!r}]"


def _slot_matches(template: str, concrete: str) -> bool:
    """Does a concrete key satisfy a templated claim?

    ``e_above_hull_{level}`` matches ``e_above_hull_emt``. A field stands for one
    identifier-like run, so ``energy_{level}`` does not match ``energy``.
    """
    if not slot_template_fields(template):
        return template == concrete
    return re.fullmatch(_template_pattern(template), concrete) is not None


def _template_pattern(template: str) -> str:
    """Regex for a slot template, with each ``{field}`` a non-empty token."""
    out, last = [], 0
    for m in _TEMPLATE.finditer(template):
        out.append(re.escape(template[last:m.start()]))
        out.append(r"[A-Za-z0-9.\-]+")
        last = m.end()
    out.append(re.escape(template[last:]))
    return "".join(out)


def _tokens(text: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", (text or "").lower()) if t}


def _overlap(a: set[str], b: set[str]) -> float:
    return float(len(a & b))


def _all_slots(mapping: Dict[str, List[str]]) -> Iterable[str]:
    for slots in mapping.values():
        yield from slots


__all__ = ["register_function", "get_registry", "FunctionRegistry",
           "AliasCollision", "CONTRACT_KEYS", "resolve_slot",
           "slot_template_fields"]
