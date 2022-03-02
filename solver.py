from typing import List, Tuple
import numpy as np
import re
import time


def toint(line):
    return int(line.strip())


def toint_tuple(line):
    line = line.strip().split(' ')
    return tuple([int(num) for num in line])


def read_file(file_name):
    with open(file_name) as f:
        num_rows = toint(f.readline())
        num_cols = toint(f.readline())
        row_clues = []
        col_clues = []

        # Read the row clues
        for _ in range(num_rows):
            row_clues.append(toint_tuple(f.readline()))
        for _ in range(num_cols):
            col_clues.append(toint_tuple(f.readline()))

    return num_rows, num_cols, row_clues, col_clues


def column_is_valid(col: np.array, clues: Tuple) -> bool:
    # Check if we have precomputed this already
    col = ''.join(col.tolist())
    if (col, clues) in valid_cols_mem:
        return valid_cols_mem[(col, clues)]

    # Given a column and the clues for that column, determine if the column follows the clues.
    original_clues = clues

    # Reverse clues list to make popping more efficient
    clues = list(clues)[::-1]
    n = len(col)

    # Split the column into consecutive sequences
    seqs = re.split('X', col)
    seqs = [seq for seq in seqs if seq]

    # Make sure the number of sequences is at most the number of clues
    if len(seqs) > len(clues):
        return False

    # If there are no clues and no sequences, then the column is valid
    if not clues:
        return True

    # Count the number of unfilled squares
    num_unfilled = n - col.find('X')

    # Check every sequence (except the ongoing one) against the clues
    for seq in seqs[:-1]:
        # The sequence length violates the clues
        if len(seq) != clues[-1]:
            return False

        clues.pop()

    # Check if the ongoing sequence violates the clues
    len_last_seq = len(seqs[-1]) if seqs else 0
    if len_last_seq > clues[-1]:
        return False

    # Check if there are enough squares left to finish clues
    num_squares_needed = clues[-1] - len_last_seq
    num_squares_needed += sum(clues[:-1]) + len(clues) - 1
    is_valid = num_squares_needed <= num_unfilled

    valid_cols_mem[(col, original_clues)] = is_valid
    return is_valid


def construct_candidates(board: np.array, ptr: Tuple[int, int], clue_idx: int) -> List[int]:
    # board: Current board state
    # ptr: Current (row, column) pointer
    # clue_idx: Index of which clue we need to fill out next
    # cur_row_clues: The list of clues for the current row
    # all_col_clues: The clues for all columns

    # Find the number of squares we need to finish the current row
    num_cols = board.shape[1]
    row, start_col = ptr
    if start_col >= num_cols:
        return []

    cur_row_clues = row_clues[row]
    num_clues = len(cur_row_clues)
    num_clues_left = num_clues - clue_idx
    num_squares_left = sum(cur_row_clues[clue_idx:]) + num_clues_left - 1

    # Find the maximum column position where we can place the next block
    clue_len = cur_row_clues[clue_idx]
    end_col = num_cols - num_squares_left

    # Create a candidate board
    board[row, start_col: end_col + clue_len] = 'O'
    is_col_valid = [False] * num_cols

    for col in range(start_col, end_col + clue_len):
        is_col_valid[col] = column_is_valid(board[:, col], col_clues[col])

    # Find the list of candidate starting columns for the current block
    candidate_cols = []
    for col in range(start_col, end_col + 1):
        if all(is_col_valid[col: col + clue_len]):
            candidate_cols.append(col)

    # Restore to original board state
    board[row, start_col: end_col + clue_len] = 'X'
    return candidate_cols


def backtrack(board: np.array, ptr: Tuple[int, int], clue_idx: int):
    if stop[0]:
        return

    num_rows = board.shape[0]
    row, col = ptr

    # Skip blank rows
    while row < num_rows and len(row_clues[row]) == 0:
        print(row, row_clues[row])
        row += 1

    if row >= num_rows:
        solution.append(board.copy())
        stop[0] = True
        return

    num_clues = len(row_clues[row])
    candidate_cols = construct_candidates(board, ptr, clue_idx)
    clue_len = row_clues[row][clue_idx]

    # Debug statements
    if debug:
        print(board)
        print(f'ptr: {ptr}, clues: {row_clues[row]}, clue_idx: {clue_idx}, cands: {candidate_cols}')

    for cand_col in candidate_cols:
        board[row, cand_col: cand_col + clue_len] = 'O'

        if clue_idx == num_clues - 1:
            next_row = row + 1
            next_col = 0
            next_clue_idx = 0
        else:
            next_row = row
            next_col = cand_col + clue_len + 1
            next_clue_idx = clue_idx + 1

        backtrack(board, (next_row, next_col), next_clue_idx)
        board[row, cand_col: cand_col + clue_len] = 'X'


def to_clues(arr):
    arr = ''.join(arr.tolist())
    seqs = re.split('X', arr)
    seqs = tuple([len(seq) for seq in seqs if seq])
    return seqs


def generate_random_puzzle(rows, cols, percent_filled):
    # Generate the board
    elems = rows * cols
    board = np.full(rows * cols, 'X')
    board[:int(elems * percent_filled)] = 'O'
    np.random.shuffle(board)
    board = board.reshape((rows, cols))

    # Generate the clues
    row_clues = [to_clues(board[i]) for i in range(rows)]
    col_clues = [to_clues(board[:, i]) for i in range(cols)]
    return board, row_clues, col_clues


if __name__ == '__main__':
    num_rows = 20
    num_cols = 20
    percent_filled = 0.6

    board_solution, row_clues, col_clues = generate_random_puzzle(num_rows, num_cols, percent_filled)
    board = np.full((num_rows, num_cols), 'X')

    # Precomputed results from column_is_valid
    debug = False
    valid_cols_mem = {}
    stop = [False]
    solution = []

    tic = time.time()
    backtrack(board, (0, 0), 0)
    toc = time.time()
    runtime = toc - tic

    print(f'Board size: {num_rows} X {num_cols}')
    print(f'Runtime: {runtime} seconds')

    print('\nRow clues:')
    for clue in row_clues:
        print(clue)

    print('\nColumn clues:')
    for clue in col_clues:
        print(clue)

    np.set_printoptions(linewidth=300)
    print()
    print(solution[0])
