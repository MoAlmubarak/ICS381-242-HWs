import numpy as np
import chess

def heuristic_chess(board):
    """
    Heuristic evaluation function for chess.
    Returns a value between -1 and 1.
    Positive values are good for white, negative values are good for black.
    """
    if board.outcome() is not None:
        # Game is over
        if board.outcome().winner == chess.WHITE:
            return 1.0  # White wins
        elif board.outcome().winner == chess.BLACK:
            return -1.0  # Black wins
        else:
            return 0.0  # Draw
    
    # Piece values (normalized)
    piece_values = {
        chess.PAWN: 0.01,
        chess.KNIGHT: 0.03,
        chess.BISHOP: 0.03,
        chess.ROOK: 0.05,
        chess.QUEEN: 0.09,
        chess.KING: 0.0  # King's value comes from checkmate
    }
    
    # Evaluate material advantage
    white_material = 0
    black_material = 0
    
    # Center control bonus (central squares are more valuable)
    center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
    white_center_control = 0
    black_center_control = 0
    
    # Count pieces and evaluate position
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            if piece.color == chess.WHITE:
                white_material += piece_values[piece.piece_type]
                
                # Bonus for controlling center
                if square in center_squares:
                    white_center_control += 0.005
                    
            else:
                black_material += piece_values[piece.piece_type]
                
                # Bonus for controlling center
                if square in center_squares:
                    black_center_control += 0.005
    
    # Add a small bonus for having more legal moves (piece mobility)
    mobility_factor = 0.0001
    
    # Store current turn
    current_turn = board.turn
    
    # Calculate white's mobility
    board.turn = chess.WHITE
    white_mobility = len(list(board.legal_moves)) * mobility_factor
    
    # Calculate black's mobility
    board.turn = chess.BLACK
    black_mobility = len(list(board.legal_moves)) * mobility_factor
    
    # Restore original turn
    board.turn = current_turn
    
    # Check bonuses
    check_bonus = 0.05
    if board.is_check():
        if board.turn == chess.BLACK:  # White is giving check
            white_material += check_bonus
        else:  # Black is giving check
            black_material += check_bonus
    
    # Combine all factors
    evaluation = (white_material - black_material) + (white_center_control - black_center_control) + (white_mobility - black_mobility)
    
    # Ensure evaluation is in range [-1, 1]
    return max(min(evaluation, 1.0), -1.0)

def is_cutoff(board, current_depth, depth_limit):
    """
    Determine if the search should be cut off at the current node.
    
    Args:
        board: The current board state
        current_depth: The current depth in the search tree
        depth_limit: The maximum depth to search
        
    Returns:
        True if search should be cut off, False otherwise
    """
    # Cut off search if maximum depth is reached or the game is over
    return current_depth >= depth_limit or board.outcome() is not None

def h_minimax(board, depth_limit):
    """
    Heuristic Minimax algorithm for chess with depth limitation.
    
    Args:
        board: A chess.Board object representing the current state
        depth_limit: Maximum depth to search
        
    Returns:
        (value, move): The utility value of the best move and the best move itself
    """
    # Start the max_node search (assuming white is maximizing)
    return max_node(board, depth_limit)

def max_node(board, depth_limit, depth=0):
    """
    Maximizer node in minimax algorithm.
    
    Args:
        board: A chess.Board object representing the current state
        depth_limit: Maximum depth to search
        depth: Current depth in the search tree
        
    Returns:
        (value, move): The utility value of the best move and the best move itself
    """
    # Check if we've reached a terminal state or depth limit
    if is_cutoff(board, depth, depth_limit):
        return heuristic_chess(board), None
    
    # Initialize best value and move
    best_value = float('-inf')
    best_move = None
    
    # Consider all legal moves
    for move in board.legal_moves:
        # Make the move on the board
        board.push(move)
        
        # Call min_node and get the value
        value, _ = min_node(board, depth_limit, depth + 1)
        
        # Undo the move
        board.pop()
        
        # Update best value and move if we found a better one
        if value > best_value:
            best_value = value
            best_move = move
    
    return best_value, best_move

def min_node(board, depth_limit, depth=0):
    """
    Minimizer node in minimax algorithm.
    
    Args:
        board: A chess.Board object representing the current state
        depth_limit: Maximum depth to search
        depth: Current depth in the search tree
        
    Returns:
        (value, move): The utility value of the best move and the best move itself
    """
    # Check if we've reached a terminal state or depth limit
    if is_cutoff(board, depth, depth_limit):
        return heuristic_chess(board), None
    
    # Initialize best value and move
    best_value = float('inf')
    best_move = None
    
    # Consider all legal moves
    for move in board.legal_moves:
        # Make the move on the board
        board.push(move)
        
        # Call max_node and get the value
        value, _ = max_node(board, depth_limit, depth + 1)
        
        # Undo the move
        board.pop()
        
        # Update best value and move if we found a better one
        if value < best_value:
            best_value = value
            best_move = move
    
    return best_value, best_move

def h_minimax_alpha_beta(board, depth_limit):
    """
    Heuristic Minimax algorithm with alpha-beta pruning for chess.
    
    Args:
        board: A chess.Board object representing the current state
        depth_limit: Maximum depth to search
        
    Returns:
        (value, move): The utility value of the best move and the best move itself
    """
    # Start the max_node search with alpha-beta pruning
    return max_node_ab(board, depth_limit, float('-inf'), float('inf'))

def max_node_ab(board, depth_limit, alpha, beta, depth=0):
    """
    Maximizer node in minimax algorithm with alpha-beta pruning.
    
    Args:
        board: A chess.Board object representing the current state
        depth_limit: Maximum depth to search
        alpha: Alpha value for pruning
        beta: Beta value for pruning
        depth: Current depth in the search tree
        
    Returns:
        (value, move): The utility value of the best move and the best move itself
    """
    # Check if we've reached a terminal state or depth limit
    if is_cutoff(board, depth, depth_limit):
        return heuristic_chess(board), None
    
    # Initialize best value and move
    best_value = float('-inf')
    best_move = None
    
    # Consider all legal moves
    for move in board.legal_moves:
        # Make the move on the board
        board.push(move)
        
        # Call min_node and get the value
        value, _ = min_node_ab(board, depth_limit, alpha, beta, depth + 1)
        
        # Undo the move
        board.pop()
        
        # Update best value and move if we found a better one
        if value > best_value:
            best_value = value
            best_move = move
        
        # Update alpha
        alpha = max(alpha, best_value)
        
        # Prune if beta <= alpha
        if beta <= alpha:
            break
    
    return best_value, best_move

def min_node_ab(board, depth_limit, alpha, beta, depth=0):
    """
    Minimizer node in minimax algorithm with alpha-beta pruning.
    
    Args:
        board: A chess.Board object representing the current state
        depth_limit: Maximum depth to search
        alpha: Alpha value for pruning
        beta: Beta value for pruning
        depth: Current depth in the search tree
        
    Returns:
        (value, move): The utility value of the best move and the best move itself
    """
    # Check if we've reached a terminal state or depth limit
    if is_cutoff(board, depth, depth_limit):
        return heuristic_chess(board), None
    
    # Initialize best value and move
    best_value = float('inf')
    best_move = None
    
    # Consider all legal moves
    for move in board.legal_moves:
        # Make the move on the board
        board.push(move)
        
        # Call max_node and get the value
        value, _ = max_node_ab(board, depth_limit, alpha, beta, depth + 1)
        
        # Undo the move
        board.pop()
        
        # Update best value and move if we found a better one
        if value < best_value:
            best_value = value
            best_move = move
        
        # Update beta
        beta = min(beta, best_value)
        
        # Prune if beta <= alpha
        if beta <= alpha:
            break
    
    return best_value, best_move