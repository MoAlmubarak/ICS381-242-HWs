import numpy as np
import chess

def heuristic_chess(board):
    """
    Evaluate the current state of the chess board.
    Returns a value between -1 and 1, where positive values favor white
    and negative values favor black.
    """
    if board.outcome() is not None:
        # If game is over, return win/loss values
        if board.outcome().winner is None:  # Draw
            return 0
        return 1 if board.outcome().winner else -1
        
    # Piece values: pawn=1, knight/bishop=3, rook=5, queen=9
    piece_values = {
        chess.PAWN: 0.01,
        chess.KNIGHT: 0.03,
        chess.BISHOP: 0.03,
        chess.ROOK: 0.05,
        chess.QUEEN: 0.09,
        chess.KING: 0  # King's value isn't used for material count
    }
    
    # Calculate material advantage
    white_material = 0
    black_material = 0
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            value = piece_values[piece.piece_type]
            if piece.color == chess.WHITE:
                white_material += value
            else:
                black_material += value
    
    # Normalize to [-1, 1] range
    total_material = white_material + black_material
    if total_material == 0:
        material_advantage = 0
    else:
        material_advantage = (white_material - black_material) / total_material
    
    # Add some bonus for center control, mobility, and other factors
    mobility_factor = 0.01
    white_moves = len(list(board.legal_moves)) if board.turn == chess.WHITE else 0
    black_moves = len(list(board.legal_moves)) if board.turn == chess.BLACK else 0
    if board.turn == chess.BLACK:  # Simulate white's moves
        board_copy = board.copy()
        board_copy.push(chess.Move.null())
        white_moves = len(list(board_copy.legal_moves))
    elif board.turn == chess.WHITE:  # Simulate black's moves
        board_copy = board.copy()
        board_copy.push(chess.Move.null())
        black_moves = len(list(board_copy.legal_moves))
    
    total_moves = white_moves + black_moves
    mobility = 0
    if total_moves > 0:
        mobility = mobility_factor * (white_moves - black_moves) / total_moves
    
    # Combine factors - material is the dominant factor
    evaluation = 0.9 * material_advantage + 0.1 * mobility
    
    # Ensure result is between -1 and 1
    return max(min(evaluation, 1), -1)

def is_cutoff(board, current_depth, depth_limit):
    """
    Determine if the search should be cut off at the current state.
    
    Args:
        board: A chess.Board object representing the current state
        current_depth: Current depth in the search tree
        depth_limit: Maximum depth to search
        
    Returns:
        True if search should be cut off, False otherwise
    """
    # Cut off if we've reached the maximum depth or if the game is over
    return current_depth >= depth_limit or board.outcome() is not None

def h_minimax(board, depth_limit):
    """
    Implementation of the h-minimax algorithm for chess.
    
    Args:
        board: A chess.Board object representing the current state
        depth_limit: Maximum depth to search
        
    Returns:
        (value, move) tuple where value is the heuristic value and move is the best move
    """
    if board.turn == chess.WHITE:
        value, move = max_node(board, depth_limit, 0)
    else:
        value, move = min_node(board, depth_limit, 0)
    
    return value, move

def max_node(board, depth_limit, current_depth):
    """Helper function for h_minimax - handles MAX player's turn"""
    # Check if we've reached a terminal state or depth limit
    if is_cutoff(board, current_depth, depth_limit):
        return heuristic_chess(board), None
    
    best_value = float('-inf')
    best_move = None
    
    for move in board.legal_moves:
        # Try this move
        board.push(move)
        
        # Recursively evaluate the position
        value, _ = min_node(board, depth_limit, current_depth + 1)
        
        # Undo the move
        board.pop()
        
        # Update best move if this is better
        if value > best_value:
            best_value = value
            best_move = move
    
    # If no moves are available (shouldn't happen in normal chess)
    if best_move is None:
        return heuristic_chess(board), None
    
    return best_value, best_move

def min_node(board, depth_limit, current_depth):
    """Helper function for h_minimax - handles MIN player's turn"""
    # Check if we've reached a terminal state or depth limit
    if is_cutoff(board, current_depth, depth_limit):
        return heuristic_chess(board), None
    
    best_value = float('inf')
    best_move = None
    
    for move in board.legal_moves:
        # Try this move
        board.push(move)
        
        # Recursively evaluate the position
        value, _ = max_node(board, depth_limit, current_depth + 1)
        
        # Undo the move
        board.pop()
        
        # Update best move if this is better (lower)
        if value < best_value:
            best_value = value
            best_move = move
    
    # If no moves are available (shouldn't happen in normal chess)
    if best_move is None:
        return heuristic_chess(board), None
    
    return best_value, best_move

def h_minimax_alpha_beta(board, depth_limit):
    """
    Implementation of the h-minimax algorithm with alpha-beta pruning for chess.
    
    Args:
        board: A chess.Board object representing the current state
        depth_limit: Maximum depth to search
        
    Returns:
        (value, move) tuple where value is the heuristic value and move is the best move
    """
    if board.turn == chess.WHITE:
        value, move = max_node_ab(board, depth_limit, 0, float('-inf'), float('inf'))
    else:
        value, move = min_node_ab(board, depth_limit, 0, float('-inf'), float('inf'))
    
    return value, move

def max_node_ab(board, depth_limit, current_depth, alpha, beta):
    """Helper function for h_minimax_alpha_beta - handles MAX player's turn with alpha-beta pruning"""
    # Check if we've reached a terminal state or depth limit
    if is_cutoff(board, current_depth, depth_limit):
        return heuristic_chess(board), None
    
    best_value = float('-inf')
    best_move = None
    
    for move in board.legal_moves:
        # Try this move
        board.push(move)
        
        # Recursively evaluate the position
        value, _ = min_node_ab(board, depth_limit, current_depth + 1, alpha, beta)
        
        # Undo the move
        board.pop()
        
        # Update best move if this is better
        if value > best_value:
            best_value = value
            best_move = move
        
        # Update alpha
        alpha = max(alpha, best_value)
        
        # Prune if we can
        if alpha >= beta:
            break
    
    # If no moves are available (shouldn't happen in normal chess)
    if best_move is None:
        return heuristic_chess(board), None
    
    return best_value, best_move

def min_node_ab(board, depth_limit, current_depth, alpha, beta):
    """Helper function for h_minimax_alpha_beta - handles MIN player's turn with alpha-beta pruning"""
    # Check if we've reached a terminal state or depth limit
    if is_cutoff(board, current_depth, depth_limit):
        return heuristic_chess(board), None
    
    best_value = float('inf')
    best_move = None
    
    for move in board.legal_moves:
        # Try this move
        board.push(move)
        
        # Recursively evaluate the position
        value, _ = max_node_ab(board, depth_limit, current_depth + 1, alpha, beta)
        
        # Undo the move
        board.pop()
        
        # Update best move if this is better (lower)
        if value < best_value:
            best_value = value
            best_move = move
        
        # Update beta
        beta = min(beta, best_value)
        
        # Prune if we can
        if alpha >= beta:
            break
    
    # If no moves are available (shouldn't happen in normal chess)
    if best_move is None:
        return heuristic_chess(board), None
    
    return best_value, best_move