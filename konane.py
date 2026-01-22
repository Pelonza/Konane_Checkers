#!/usr/bin/env python3
"""
Konane (Hawaiian Checkers) — text-based engine + agents (Human / Random / Greedy / Minimax w/ Alpha-Beta)

Rules implemented (common classroom/standard variant):
- Initial board is alternating Black/White.
- Turn 1 (Black): remove ANY one of your stones (creates the first empty).
- Turn 2 (White): remove ONE of your stones that is orthogonally adjacent to the empty.
- Thereafter: moves are one or more orthogonal jumps:
    * Jump over an adjacent opponent stone into an empty square two steps away.
    * The jumped opponent stone is removed.
    * Multi-jumps are allowed/required to be represented as a sequence; the same piece continues jumping.
- Game ends when the current player has no legal moves; that player loses.

CLI supports:
- Human vs AI
- AI vs AI
- Self-play tournaments (repeat games)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Iterable
import random
import time
import math
import argparse
import sys

# ----------------------------
# Types / Constants
# ----------------------------

Coord = Tuple[int, int]

EMPTY = "."
BLACK = "B"
WHITE = "W"

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # N,S,W,E


def opponent(player: str) -> str:
    return WHITE if player == BLACK else BLACK


# ----------------------------
# Move representation
# ----------------------------

@dataclass(frozen=True)
class Move:
    """
    Konane move:
    - For opening removals: path has length 1 (just the removed coordinate).
    - For jump moves: path is the sequence of landing squares, starting with the start square.
      Example: start at (2,1) jump to (2,3) then to (4,3):
        path = [(2,1), (2,3), (4,3)]
    """
    kind: str  # "remove" or "jump"
    path: Tuple[Coord, ...]  # immutable for hashing

    def __str__(self) -> str:
        if self.kind == "remove":
            r, c = self.path[0]
            return f"REMOVE ({r},{c})"
        else:
            return " -> ".join(f"({r},{c})" for (r, c) in self.path)


# ----------------------------
# Game State / Engine
# ----------------------------

class GameState:
    def __init__(self, n: int = 8, board: Optional[List[List[str]]] = None,
                 current_player: str = BLACK, turn_number: int = 1):
        self.n = n
        self.board = board if board is not None else self._initial_board(n)
        self.current_player = current_player
        self.turn_number = turn_number  # 1-based
        # For opening constraint: remember the first empty made by Black removal
        self.opening_empty: Optional[Coord] = self._find_opening_empty_if_any()

    @staticmethod
    def _initial_board(n: int) -> List[List[str]]:
        board = [[EMPTY] * n for _ in range(n)]
        for r in range(n):
            for c in range(n):
                board[r][c] = BLACK if (r + c) % 2 == 0 else WHITE
        return board

    def clone(self) -> "GameState":
        return GameState(
            n=self.n,
            board=[row[:] for row in self.board],
            current_player=self.current_player,
            turn_number=self.turn_number
        )

    def _find_opening_empty_if_any(self) -> Optional[Coord]:
        # During the first two turns there will be 0 or 1 empty.
        # After that there can be many empties; we keep opening_empty only for turn 2 legality.
        empties = [(r, c) for r in range(self.n) for c in range(self.n) if self.board[r][c] == EMPTY]
        if self.turn_number == 2:
            # must be exactly one empty after black's opening removal
            return empties[0] if len(empties) == 1 else None
        return None

    # ---------
    # Printing
    # ---------

    def pretty(self) -> str:
        lines = []
        header = "    " + " ".join(f"{c:2d}" for c in range(self.n))
        lines.append(header)
        lines.append("    " + "--" * self.n)
        for r in range(self.n):
            lines.append(f"{r:2d} | " + " ".join(f"{self.board[r][c]:2s}" for c in range(self.n)))
        return "\n".join(lines)

    # -------------
    # Hash / Keying
    # -------------

    def key(self) -> Tuple:
        # Include current_player and turn_number phase (opening matters for move gen)
        flat = tuple(cell for row in self.board for cell in row)
        return (self.n, flat, self.current_player, self.turn_number)

    # -----------------
    # Rule primitives
    # -----------------

    def inside(self, r: int, c: int) -> bool:
        return 0 <= r < self.n and 0 <= c < self.n

    def get(self, rc: Coord) -> str:
        r, c = rc
        return self.board[r][c]

    def set(self, rc: Coord, val: str) -> None:
        r, c = rc
        self.board[r][c] = val

    def count_pieces(self, player: str) -> int:
        return sum(1 for r in range(self.n) for c in range(self.n) if self.board[r][c] == player)

    def empties(self) -> List[Coord]:
        return [(r, c) for r in range(self.n) for c in range(self.n) if self.board[r][c] == EMPTY]

    # -----------------
    # Move generation
    # -----------------

    def legal_moves(self) -> List[Move]:
        if self.turn_number == 1:
            return self._legal_opening_black_removals()
        elif self.turn_number == 2:
            return self._legal_opening_white_removals()
        else:
            return self._legal_jump_moves(self.current_player)

    def _legal_opening_black_removals(self) -> List[Move]:
        # Black removes any one BLACK stone
        moves: List[Move] = []
        if self.current_player != BLACK:
            return moves
        for r in range(self.n):
            for c in range(self.n):
                if self.board[r][c] == BLACK:
                    moves.append(Move("remove", ((r, c),)))
        return moves

    def _legal_opening_white_removals(self) -> List[Move]:
        # White removes a WHITE stone orthogonally adjacent to the single empty created by Black.
        moves: List[Move] = []
        if self.current_player != WHITE:
            return moves
        empties = self.empties()
        if len(empties) != 1:
            return moves
        er, ec = empties[0]
        for dr, dc in DIRS:
            nr, nc = er + dr, ec + dc
            if self.inside(nr, nc) and self.board[nr][nc] == WHITE:
                moves.append(Move("remove", ((nr, nc),)))
        return moves

    def _legal_jump_moves(self, player: str) -> List[Move]:
        moves: List[Move] = []
        opp = opponent(player)

        for r in range(self.n):
            for c in range(self.n):
                if self.board[r][c] != player:
                    continue
                start = (r, c)
                # find all multi-jump sequences starting from start
                self._dfs_jumps_from(start, player, opp, [(r, c)], moves)
        return moves

    def _dfs_jumps_from(
        self,
        start: Coord,
        player: str,
        opp: str,
        path: List[Coord],
        out_moves: List[Move]
    ) -> None:
        """
        DFS over jump sequences. `path` is list of landing squares including initial position.
        If at least one jump is made, each complete path is a legal move.
        """
        r, c = path[-1]
        extended = False

        for dr, dc in DIRS:
            mid = (r + dr, c + dc)
            land = (r + 2 * dr, c + 2 * dc)
            if not (self.inside(*mid) and self.inside(*land)):
                continue
            if self.get(mid) == opp and self.get(land) == EMPTY:
                # Try this jump by applying it temporarily
                extended = True
                # Apply jump
                jumped = mid
                from_sq = (r, c)
                to_sq = land
                self.set(from_sq, EMPTY)
                self.set(jumped, EMPTY)
                self.set(to_sq, player)

                path.append(to_sq)
                # Continue jumping from new position
                self._dfs_jumps_from(start, player, opp, path, out_moves)
                path.pop()

                # Undo jump
                self.set(to_sq, EMPTY)
                self.set(jumped, opp)
                self.set(from_sq, player)

        # If no further extension and we have made at least one jump, record the move
        if not extended and len(path) >= 2:
            out_moves.append(Move("jump", tuple(path)))

    # -----------------
    # Apply moves
    # -----------------

    def apply(self, move: Move) -> "GameState":
        ns = self.clone()
        player = ns.current_player
        opp = opponent(player)

        if move.kind == "remove":
            (r, c) = move.path[0]
            if ns.board[r][c] != player:
                raise ValueError("Illegal removal: not your piece")
            ns.board[r][c] = EMPTY

            # Update opening tracking
            if ns.turn_number == 1:
                # after black removal, exactly one empty should exist
                ns.opening_empty = (r, c)
            elif ns.turn_number == 2:
                ns.opening_empty = None  # no longer needed

        elif move.kind == "jump":
            path = list(move.path)
            if len(path) < 2:
                raise ValueError("Illegal jump: empty path")
            (sr, sc) = path[0]
            if ns.board[sr][sc] != player:
                raise ValueError("Illegal jump: start not your piece")

            # execute each jump segment
            for i in range(len(path) - 1):
                (r1, c1) = path[i]
                (r2, c2) = path[i + 1]
                dr = r2 - r1
                dc = c2 - c1
                if (abs(dr), abs(dc)) not in [(2, 0), (0, 2)]:
                    raise ValueError("Illegal jump: must move by 2 orthogonally")
                mr, mc = (r1 + r2) // 2, (c1 + c2) // 2  # jumped square
                if ns.board[mr][mc] != opp or ns.board[r2][c2] != EMPTY:
                    raise ValueError("Illegal jump: missing opponent or landing not empty")

                # move piece and remove jumped
                ns.board[r1][c1] = EMPTY
                ns.board[mr][mc] = EMPTY
                ns.board[r2][c2] = player

        else:
            raise ValueError("Unknown move kind")

        # advance turn
        ns.turn_number += 1
        ns.current_player = opponent(ns.current_player)

        return ns

    # -----------------
    # Terminal condition
    # -----------------

    def is_terminal(self) -> bool:
        # terminal AFTER opening once regular play begins, but even in opening if no legal moves
        return len(self.legal_moves()) == 0


# ----------------------------
# Evaluation (Heuristics)
# ----------------------------

class Evaluator:
    def evaluate(self, state: GameState, perspective: str) -> float:
        raise NotImplementedError


class MobilityEvaluator(Evaluator):
    """
    Classic starter heuristic:
      score = mobility(perspective) - mobility(opponent)
    """

    def evaluate(self, state: GameState, perspective: str) -> float:
        if state.is_terminal():
            # If it's terminal for current player, that player loses.
            # Determine winner relative to perspective.
            current = state.current_player
            loser = current
            winner = opponent(loser)
            return 1e9 if winner == perspective else -1e9

        # mobility is expensive if we generate all moves, but acceptable for class sizes/depths
        saved_player = state.current_player

        state.current_player = perspective
        mp = len(state.legal_moves())

        state.current_player = opponent(perspective)
        mo = len(state.legal_moves())

        state.current_player = saved_player
        return float(mp - mo)


# ----------------------------
# Agents
# ----------------------------

class Agent:
    name: str = "Agent"
    def get_move(self, state: GameState) -> Move:
        raise NotImplementedError


class HumanAgent(Agent):
    name = "Human"

    def get_move(self, state: GameState) -> Move:
        moves = state.legal_moves()
        if not moves:
            raise RuntimeError("No legal moves")

        print("\nLegal moves:")
        # Show a few if many
        max_show = 25
        for i, m in enumerate(moves[:max_show]):
            print(f"  [{i}] {m}")
        if len(moves) > max_show:
            print(f"  ... ({len(moves) - max_show} more)")

        while True:
            raw = input(
                "\nEnter move:\n"
                "  Opening remove:  r,c\n"
                "  Jump sequence:   r1,c1 -> r2,c2 -> r3,c3 ...\n"
                "  Quit: q"
                "> "
            ).strip()

            try:
                mv = parse_move(raw)
            except Exception as e:
                print(f"Could not parse: {e}")
                continue

            # Validate by exact match against legal moves
            for legal in moves:
                if legal == mv:
                    return legal

            print("That move is not legal from this position. Try again.")


class RandomAgent(Agent):
    name = "Random"
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def get_move(self, state: GameState) -> Move:
        moves = state.legal_moves()
        if not moves:
            raise RuntimeError("No legal moves")
        return self.rng.choice(moves)


class GreedyMobilityAgent(Agent):
    """
    One-ply greedy: choose the move that maximizes mobility difference after the move.
    """
    name = "GreedyMobility"

    def __init__(self, evaluator: Optional[Evaluator] = None):
        self.evaluator = evaluator or MobilityEvaluator()

    def get_move(self, state: GameState) -> Move:
        moves = state.legal_moves()
        if not moves:
            raise RuntimeError("No legal moves")
        me = state.current_player
        best = None
        best_score = -math.inf
        for m in moves:
            ns = state.apply(m)
            score = self.evaluator.evaluate(ns, me)
            if score > best_score:
                best_score = score
                best = m
        return best  # type: ignore


class MinimaxAlphaBetaAgent(Agent):
    name = "MinimaxAB"

    def __init__(
        self,
        depth: int = 4,
        evaluator: Optional[Evaluator] = None,
        use_tt: bool = True,
        time_limit_s: Optional[float] = None,
        move_ordering: bool = True
    ):
        self.depth = depth
        self.evaluator = evaluator or MobilityEvaluator()
        self.use_tt = use_tt
        self.time_limit_s = time_limit_s
        self.move_ordering = move_ordering
        self._tt: Dict[Tuple, Tuple[int, float]] = {}  # key -> (depth, score)
        self._start_time = 0.0

    def get_move(self, state: GameState) -> Move:
        moves = state.legal_moves()
        if not moves:
            raise RuntimeError("No legal moves")

        self._start_time = time.time()
        me = state.current_player

        best_score = -math.inf
        best_move = moves[0]

        ordered = moves
        if self.move_ordering:
            ordered = self._order_moves(state, moves, me)

        alpha, beta = -math.inf, math.inf

        for m in ordered:
            if self._timed_out():
                break
            ns = state.apply(m)
            score = self._min_value(ns, self.depth - 1, alpha, beta, me)
            if score > best_score:
                best_score = score
                best_move = m
            alpha = max(alpha, best_score)

        return best_move

    def _timed_out(self) -> bool:
        return self.time_limit_s is not None and (time.time() - self._start_time) >= self.time_limit_s

    def _order_moves(self, state: GameState, moves: List[Move], perspective: str) -> List[Move]:
        # Simple ordering: evaluate shallowly so alpha-beta prunes more.
        scored = []
        for m in moves:
            ns = state.apply(m)
            scored.append((self.evaluator.evaluate(ns, perspective), m))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored]

    def _tt_lookup(self, state: GameState, depth: int) -> Optional[float]:
        if not self.use_tt:
            return None
        k = state.key()
        if k in self._tt:
            stored_depth, stored_score = self._tt[k]
            if stored_depth >= depth:
                return stored_score
        return None

    def _tt_store(self, state: GameState, depth: int, score: float) -> None:
        if not self.use_tt:
            return
        k = state.key()
        prev = self._tt.get(k)
        if prev is None or prev[0] < depth:
            self._tt[k] = (depth, score)

    def _max_value(self, state: GameState, depth: int, alpha: float, beta: float, perspective: str) -> float:
        if self._timed_out():
            return self.evaluator.evaluate(state, perspective)
        if depth == 0 or state.is_terminal():
            return self.evaluator.evaluate(state, perspective)

        tt = self._tt_lookup(state, depth)
        if tt is not None:
            return tt

        v = -math.inf
        moves = state.legal_moves()
        if self.move_ordering:
            moves = self._order_moves(state, moves, perspective)

        for m in moves:
            ns = state.apply(m)
            v = max(v, self._min_value(ns, depth - 1, alpha, beta, perspective))
            alpha = max(alpha, v)
            if alpha >= beta:
                break

        self._tt_store(state, depth, v)
        return v

    def _min_value(self, state: GameState, depth: int, alpha: float, beta: float, perspective: str) -> float:
        if self._timed_out():
            return self.evaluator.evaluate(state, perspective)
        if depth == 0 or state.is_terminal():
            return self.evaluator.evaluate(state, perspective)

        tt = self._tt_lookup(state, depth)
        if tt is not None:
            return tt

        v = math.inf
        moves = state.legal_moves()
        if self.move_ordering:
            # For MIN nodes, ordering by perspective eval still helps often; could reverse, but keep simple.
            moves = self._order_moves(state, moves, perspective)

        for m in moves:
            ns = state.apply(m)
            v = min(v, self._max_value(ns, depth - 1, alpha, beta, perspective))
            beta = min(beta, v)
            if beta <= alpha:
                break

        self._tt_store(state, depth, v)
        return v


# ----------------------------
# Parsing & Utilities
# ----------------------------

def parse_coord(token: str) -> Coord:
    token = token.strip()
    if "," not in token:
        raise ValueError("Expected 'r,c'")
    r_s, c_s = token.split(",", 1)
    return int(r_s.strip()), int(c_s.strip())


def parse_move(s: str) -> Move:
    s = s.strip()
    if "q" in s:
        sys.exit()
    if "->" in s:
        parts = [p.strip() for p in s.split("->")]
        coords = tuple(parse_coord(p) for p in parts)
        if len(coords) < 2:
            raise ValueError("Jump move must have at least 2 coordinates")
        return Move("jump", coords)
    else:
        # removal
        rc = parse_coord(s)
        return Move("remove", (rc,))


# ----------------------------
# Game Loop / Controller
# ----------------------------

def play_game(
    n: int,
    black_agent: Agent,
    white_agent: Agent,
    seed: Optional[int] = None,
    print_every_turn: bool = True,
    max_turns: int = 10_000
) -> str:
    if seed is not None:
        random.seed(seed)

    state = GameState(n=n)
    agents = {BLACK: black_agent, WHITE: white_agent}

    if print_every_turn:
        print("\nInitial board:")
        print(state.pretty())
        print(f"\nTurn {state.turn_number}: {state.current_player} to move ({agents[state.current_player].name})")

    for _ in range(max_turns):
        moves = state.legal_moves()
        if not moves:
            loser = state.current_player
            winner = opponent(loser)
            if print_every_turn:
                print(f"\nNo legal moves for {loser}. {winner} wins!")
            return winner

        agent = agents[state.current_player]
        move = agent.get_move(state)

        if print_every_turn:
            print(f"\n{state.current_player} ({agent.name}) plays: {move}")

        state = state.apply(move)

        if print_every_turn:
            print(state.pretty())
            print(f"\nTurn {state.turn_number}: {state.current_player} to move ({agents[state.current_player].name})")

    raise RuntimeError("Exceeded max_turns—likely a bug (Konane should always finish).")


def tournament(
    games: int,
    n: int,
    black_agent: Agent,
    white_agent: Agent,
    seed: Optional[int] = None
) -> None:
    wins = {BLACK: 0, WHITE: 0}
    for i in range(games):
        # Alternate who goes first by swapping agent colors if desired; for now keep fixed.
        winner = play_game(
            n=n,
            black_agent=black_agent,
            white_agent=white_agent,
            seed=(None if seed is None else seed + i),
            print_every_turn=False
        )
        wins[winner] += 1
    print(f"\nTournament results ({games} games) on {n}x{n}:")
    print(f"  Black ({black_agent.name}) wins: {wins[BLACK]}")
    print(f"  White ({white_agent.name}) wins: {wins[WHITE]}")


# ----------------------------
# CLI Wiring
# ----------------------------

def build_agent(spec: str) -> Agent:
    """
    Agent spec examples:
      human
      random
      greedy
      minimax:4
      minimax:5:time=1.0
    """
    spec = spec.strip().lower()
    if spec == "human":
        return HumanAgent()
    if spec == "random":
        return RandomAgent()
    if spec == "greedy":
        return GreedyMobilityAgent()

    if spec.startswith("minimax"):
        # parse depth and optional time
        # formats:
        #   minimax
        #   minimax:4
        #   minimax:4:time=1.0
        parts = spec.split(":")
        depth = 4
        time_limit = None
        if len(parts) >= 2 and parts[1].isdigit():
            depth = int(parts[1])
        if len(parts) >= 3 and parts[2].startswith("time="):
            time_limit = float(parts[2].split("=", 1)[1])
        return MinimaxAlphaBetaAgent(depth=depth, time_limit_s=time_limit)

    raise ValueError(f"Unknown agent spec: {spec}")


def main():
    parser = argparse.ArgumentParser(description="Konane: text-based + minimax alpha-beta")
    parser.add_argument("--n", type=int, default=8, help="Board size (NxN), usually even.")
    parser.add_argument("--black", type=str, default="human", help="Black agent: human|random|greedy|minimax[:depth[:time=sec]]")
    parser.add_argument("--white", type=str, default="minimax:4", help="White agent: human|random|greedy|minimax[:depth[:time=sec]]")
    parser.add_argument("--tournament", type=int, default=0, help="Run N games without printing boards.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    args = parser.parse_args()

    black_agent = build_agent(args.black)
    white_agent = build_agent(args.white)

    if args.tournament and args.tournament > 0:
        tournament(args.tournament, args.n, black_agent, white_agent, seed=args.seed)
    else:
        winner = play_game(args.n, black_agent, white_agent, seed=args.seed, print_every_turn=True)
        print(f"\nWinner: {winner}")


if __name__ == "__main__":
    main()
