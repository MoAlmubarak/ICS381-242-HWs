import chess
import numpy as np

def heuristic_chess(board):
    """
    Calculate the heuristic value of a chess board from the WHITE player's perspective.
    
    For terminal states:
    - White wins: +1
    - Black wins: -1
    - Draw: 0
    
    For non-terminal states:
    - Weighted sum of piece differences, scaled to range [-1, 1]
    
    Args:
        board: A chess.Board object representing the current board state
        
    Returns:
        float: Heuristic value of the board state
    """
    # Check if the game is over (terminal state)
    outcome = board.outcome()
    if outcome is not None:
        winner = outcome.winner
        
        if outcome.termination == chess.Termination.CHECKMATE:
            if winner:  # White wins
                return 1.0
            else:  # Black wins
                return -1.0
        else:  # Draw (stalemate, insufficient material, etc.)
            return 0.0
    
    # Non-terminal state - compute weighted piece differences
    # Count the number of pieces for each side
    white_pawns = len(board.pieces(chess.PAWN, chess.WHITE))
    black_pawns = len(board.pieces(chess.PAWN, chess.BLACK))
    white_knights = len(board.pieces(chess.KNIGHT, chess.WHITE))
    black_knights = len(board.pieces(chess.KNIGHT, chess.BLACK))
    white_bishops = len(board.pieces(chess.BISHOP, chess.WHITE))
    black_bishops = len(board.pieces(chess.BISHOP, chess.BLACK))
    white_rooks = len(board.pieces(chess.ROOK, chess.WHITE))
    black_rooks = len(board.pieces(chess.ROOK, chess.BLACK))
    white_queens = len(board.pieces(chess.QUEEN, chess.WHITE))
    black_queens = len(board.pieces(chess.QUEEN, chess.BLACK))
    
    # Calculate feature differences
    f_pawn = white_pawns - black_pawns
    f_knight = white_knights - black_knights
    f_bishop = white_bishops - black_bishops
    f_rook = white_rooks - black_rooks
    f_queen = white_queens - black_queens
    
    # Calculate weighted sum and scale
    heuristic_value = (1.25 * f_pawn + 3.5 * f_knight + 5 * f_bishop + 5 * f_rook + 10 * f_queen) / 100
    
    return heuristic_value

def is_cutoff(board, current_depth, depth_limit=2):
    """
    Determine whether to cut off the search at the current node.
    
    Args:
        board: A chess.Board object representing the current board state
        current_depth: The current depth in the game tree
        depth_limit: The maximum depth to search
        
    Returns:
        bool: True if search should be cut off, False otherwise
    """
    # Check if the game is over (terminal state)
    if board.outcome() is not None:
        return True
    
    # Check if we've reached the depth limit
    if current_depth >= depth_limit:
        return True
    
    # Otherwise, continue searching
    return False

def h_minimax(board, depth_limit=2):
    """
    Perform heuristic minimax search to find the best move for the current player.
    
    Args:
        board: A chess.Board object representing the current board state
        depth_limit: The maximum depth to search
        
    Returns:
        tuple: (heuristic_value, move) - The heuristic value of the best move and the best move itself
    """
    def max_node(board, current_depth, depth_limit):
        """Helper function to implement MAX node behavior."""
        # Check if cutoff
        if is_cutoff(board, current_depth, depth_limit):
            return heuristic_chess(board), None
        
        v = -np.inf
        move = None
        
        # Try all possible moves
        for a in board.legal_moves:
            # Make the move
            board.push(a)
            
            # Get value from MIN node
            v2, _ = min_node(board, current_depth + 1, depth_limit)
            
            # Undo the move
            board.pop()
            
            # Update best value and move
            if v2 > v:
                v = v2
                move = a
        
        return v, move
    
    def min_node(board, current_depth, depth_limit):
        """Helper function to implement MIN node behavior."""
        # Check if cutoff
        if is_cutoff(board, current_depth, depth_limit):
            return heuristic_chess(board), None
        
        v = np.inf
        move = None
        
        # Try all possible moves
        for a in board.legal_moves:
            # Make the move
            board.push(a)
            
            # Get value from MAX node
            v2, _ = max_node(board, current_depth + 1, depth_limit)
            
            # Undo the move
            board.pop()
            
            # Update best value and move
            if v2 < v:
                v = v2
                move = a
        
        return v, move
    
    # Start the search from the appropriate node based on current player
    if board.turn == chess.WHITE:  # If it's White's turn (MAX)
        return max_node(board, 0, depth_limit)
    else:  # If it's Black's turn (MIN)
        return min_node(board, 0, depth_limit)

def h_minimax_alpha_beta(board, depth_limit=2):
    """
    Perform heuristic minimax search with alpha-beta pruning to find the best move for the current player.
    
    Args:
        board: A chess.Board object representing the current board state
        depth_limit: The maximum depth to search
        
    Returns:
        tuple: (heuristic_value, move) - The heuristic value of the best move and the best move itself
    """
    def max_node_ab(board, current_depth, depth_limit, alpha, beta):
        """Helper function to implement MAX node behavior with alpha-beta pruning."""
        # Check if cutoff
        if is_cutoff(board, current_depth, depth_limit):
            return heuristic_chess(board), None
        
        v = -np.inf
        move = None
        
        # Try all possible moves
        for a in board.legal_moves:
            # Make the move
            board.push(a)
            
            # Get value from MIN node
            v2, _ = min_node_ab(board, current_depth + 1, depth_limit, alpha, beta)
            
            # Undo the move
            board.pop()
            
            # Update best value and move
            if v2 > v:
                v = v2
                move = a
                
            # Update alpha
            alpha = max(alpha, v)
            
            # Alpha-beta pruning
            if v >= beta:
                return v, move
        
        return v, move
    
    def min_node_ab(board, current_depth, depth_limit, alpha, beta):
        """Helper function to implement MIN node behavior with alpha-beta pruning."""
        # Check if cutoff
        if is_cutoff(board, current_depth, depth_limit):
            return heuristic_chess(board), None
        
        v = np.inf
        move = None
        
        # Try all possible moves
        for a in board.legal_moves:
            # Make the move
            board.push(a)
            
            # Get value from MAX node
            v2, _ = max_node_ab(board, current_depth + 1, depth_limit, alpha, beta)
            
            # Undo the move
            board.pop()
            
            # Update best value and move
            if v2 < v:
                v = v2
                move = a
                
            # Update beta
            beta = min(beta, v)
            
            # Alpha-beta pruning
            if v <= alpha:
                return v, move
        
        return v, move
    
    # Start the search from the appropriate node based on current player
    if board.turn == chess.WHITE:  # If it's White's turn (MAX)
        return max_node_ab(board, 0, depth_limit, -np.inf, np.inf)
    else:  # If it's Black's turn (MIN)
        return min_node_ab(board, 0, depth_limit, -np.inf, np.inf)
