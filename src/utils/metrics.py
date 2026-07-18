from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


OpKind = Literal["equal", "substitution", "deletion", "insertion"]


@dataclass(frozen=True)
class EditOp:
    kind: OpKind
    ref: str | None
    hyp: str | None


@dataclass(frozen=True)
class EditStats:
    substitutions: int
    deletions: int
    insertions: int
    ref_words: int

    @property
    def wer(self) -> float:
        return 0.0 if self.ref_words == 0 else (
            self.substitutions + self.deletions + self.insertions
        ) / self.ref_words


def edit_stats(ref: str, hyp: str) -> EditStats:
    ops = align_words(ref, hyp)
    return EditStats(
        substitutions=sum(1 for op in ops if op.kind == "substitution"),
        deletions=sum(1 for op in ops if op.kind == "deletion"),
        insertions=sum(1 for op in ops if op.kind == "insertion"),
        ref_words=len(ref.split()),
    )


def align_words(ref: str, hyp: str) -> list[EditOp]:
    r = ref.split()
    h = hyp.split()
    dp = [[0 for _ in range(len(h) + 1)] for _ in range(len(r) + 1)]
    back: list[list[tuple[int, int, OpKind]]] = [
        [(0, 0, "equal") for _ in range(len(h) + 1)] for _ in range(len(r) + 1)
    ]
    for i in range(1, len(r) + 1):
        dp[i][0] = i
        back[i][0] = (i - 1, 0, "deletion")
    for j in range(1, len(h) + 1):
        dp[0][j] = j
        back[0][j] = (0, j - 1, "insertion")
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                candidates = [(dp[i - 1][j - 1], i - 1, j - 1, "equal")]
            else:
                candidates = [(dp[i - 1][j - 1] + 1, i - 1, j - 1, "substitution")]
            candidates.extend([
                (dp[i - 1][j] + 1, i - 1, j, "deletion"),
                (dp[i][j - 1] + 1, i, j - 1, "insertion"),
            ])
            # Prefer exact matches, then substitutions, then deletions, then insertions
            # for deterministic S/D/I attribution when edit distance ties.
            rank = {"equal": 0, "substitution": 1, "deletion": 2, "insertion": 3}
            cost, pi, pj, kind = min(candidates, key=lambda x: (x[0], rank[x[3]]))
            dp[i][j] = cost
            back[i][j] = (pi, pj, kind)  # type: ignore[assignment]
    ops: list[EditOp] = []
    i, j = len(r), len(h)
    while i > 0 or j > 0:
        pi, pj, kind = back[i][j]
        if kind == "equal":
            ops.append(EditOp(kind, r[i - 1], h[j - 1]))
        elif kind == "substitution":
            ops.append(EditOp(kind, r[i - 1], h[j - 1]))
        elif kind == "deletion":
            ops.append(EditOp(kind, r[i - 1], None))
        elif kind == "insertion":
            ops.append(EditOp(kind, None, h[j - 1]))
        i, j = pi, pj
    ops.reverse()
    return ops
