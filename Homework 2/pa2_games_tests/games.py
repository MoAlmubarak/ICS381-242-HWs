import chess
import numpy as np


class Profiler:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.h_minimax_calls = 0
        self.max_node_calls = 0
        self.min_node_calls = 0
    
    def increment_h_minimax(self):
        self.h_minimax_calls += 1
    
    def increment_max_node(self):
        self.max_node_calls += 1
    
    def increment_min_node(self):
        self.min_node_calls += 1

# Create a global profiler instance
profiler = Profiler()

# Piece values as constants
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0
}
MAX_MATERIAL = 39  # Maximum possible material value per side

def heuristic_chess(board):
    """
    Evaluate the chess board state with a value between -1 and 1.
    Positive values favor white, negative values favor black.
    """
    # Check for game over conditions first
    outcome = board.outcome()
    if outcome is not None:
        if outcome.winner is None:  # Draw
            return 0
        return 1 if outcome.winner else -1

    # Calculate material advantage efficiently
    white_material = 0
    black_material = 0
    
    for square, piece in board.piece_map().items():
        value = PIECE_VALUES[piece.piece_type]
        if piece.color == chess.WHITE:
            white_material += value
        else:
            black_material += value

    # Normalize to [-0.5, 0.5]
    material_advantage = (white_material - black_material) / (2 * MAX_MATERIAL)
    
    return material_advantage

def is_cutoff(board, current_depth, depth_limit):
    """Determine if search should stop at this node."""
    return current_depth >= depth_limit or board.is_game_over()

def h_minimax(board, depth_limit):
    """H-minimax algorithm for chess."""
    profiler.reset()
    profiler.increment_h_minimax()
    
    if board.turn == chess.WHITE:
        value, best_move = max_node(board, depth_limit, 0)
    else:
        value, best_move = min_node(board, depth_limit, 0)
    
    return value, best_move

def max_node(board, depth_limit, current_depth):
    """Maximizing player in minimax."""
    profiler.increment_max_node()
    
    if is_cutoff(board, current_depth, depth_limit):
        return heuristic_chess(board), None

    best_value = float('-inf')
    best_move = None
    
    # Sort moves to improve pruning chances
    legal_moves = list(board.legal_moves)
    
    for move in legal_moves:
        board.push(move)
        value, _ = min_node(board, depth_limit, current_depth + 1)
        board.pop()

        if value > best_value:
            best_value = value
            best_move = move
            
    return best_value, best_move

def min_node(board, depth_limit, current_depth):
    """Minimizing player in minimax."""
    profiler.increment_min_node()
    
    if is_cutoff(board, current_depth, depth_limit):
        return heuristic_chess(board), None

    best_value = float('inf')
    best_move = None
    
    legal_moves = list(board.legal_moves)
    
    for move in legal_moves:
        board.push(move)
        value, _ = max_node(board, depth_limit, current_depth + 1)
        board.pop()

        if value < best_value:
            best_value = value
            best_move = move
            
    return best_value, best_move

def h_minimax_alpha_beta(board, depth_limit):
    """H-minimax with alpha-beta pruning for chess."""
    profiler.reset()
    profiler.increment_h_minimax()
    
    if board.turn == chess.WHITE:
        value, best_move = max_node_ab(board, depth_limit, 0, float('-inf'), float('inf'))
    else:
        value, best_move = min_node_ab(board, depth_limit, 0, float('-inf'), float('inf'))
    
    return value, best_move

def max_node_ab(board, depth_limit, current_depth, alpha, beta):
    """Maximizing player with alpha-beta pruning."""
    profiler.increment_max_node()
    
    if is_cutoff(board, current_depth, depth_limit):
        return heuristic_chess(board), None

    best_value = float('-inf')
    best_move = None
    
    legal_moves = list(board.legal_moves)
    
    for move in legal_moves:
        board.push(move)
        value, _ = min_node_ab(board, depth_limit, current_depth + 1, alpha, beta)
        board.pop()

        if value > best_value:
            best_value = value
            best_move = move
            
        alpha = max(alpha, best_value)
        if alpha >= beta:  # Beta cutoff
            break

    return best_value, best_move

def min_node_ab(board, depth_limit, current_depth, alpha, beta):
    """Minimizing player with alpha-beta pruning."""
    profiler.increment_min_node()
    
    if is_cutoff(board, current_depth, depth_limit):
        return heuristic_chess(board), None

    best_value = float('inf')
    best_move = None

    legal_moves = list(board.legal_moves)
    
    for move in legal_moves:
        board.push(move)
        value, _ = max_node_ab(board, depth_limit, current_depth + 1, alpha, beta)
        board.pop()

        if value < best_value:
            best_value = value
            best_move = move

        beta = min(beta, best_value)
        if alpha >= beta:  # Alpha cutoff
            break

    return best_value, best_move
