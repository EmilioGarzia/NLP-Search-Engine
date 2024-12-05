# Spelling correction module 
#
# @author Emilio Garzia, 2024

import numpy as np

############################################################ LEVENSHTEIN DISTANCE ############################################################
class levenshtein:
    """
    A class to calculate the Levenshtein distance (edit distance) between two strings.
    This implementation also tracks the operations required to transform one string
    into another and supports custom costs for insertions, deletions, and replacements.

    Attributes:
        source (str): The source string.
        target (str): The target string.
        replace_cost (int): The cost of a replacement operation.
        insert_cost (int): The cost of an insertion operation.
        delete_cost (int): The cost of a deletion operation.
        distance_matrix (np.ndarray): Matrix storing the edit distances.
        backtrace_matrix (np.ndarray): Matrix storing the operations performed.
    """
    def __init__(self, source, target, replace_cost=1, insert_cost=1, delete_cost=1):
        """
        Initializes the levenshtein object and builds the edit distance matrix.

        Args:
            source (str): The source string.
            target (str): The target string.
            replace_cost (int, optional): The cost of replacing a character. Default is 1.
            insert_cost (int, optional): The cost of inserting a character. Default is 1.
            delete_cost (int, optional): The cost of deleting a character. Default is 1.
        """
        self.source = source.lower()
        self.target = target.lower()
        self.M = len(source)
        self.N = len(target)
        self.replace_cost = replace_cost
        self.insert_cost = insert_cost
        self.delete_cost = delete_cost
        self.distance_matrix = np.zeros(shape=(self.M+1, self.N+1), dtype=np.uint8)
        self.backtrace_matrix = np.zeros(shape=(self.M+1, self.N+1), dtype=np.uint8)
               
        self.build_levenshtein()

    def build_levenshtein(self):
        """
        Builds the distance and backtrace matrices for the Levenshtein algorithm.
        """
        # Matrices initialization
        for i in range((self.M)+1):
            self.distance_matrix[i][0] = i
            self.backtrace_matrix[i][0] = self.DELETE
        for j in range((self.N)+1):
            self.distance_matrix[0][j] = j
            self.backtrace_matrix[0][j] = self.INSERT
        
        # Matrices calculation 
        for i in range(1, (self.M)+1):
            for j in range(1, (self.N)+1):
                if self.source[i-1] == self.target[j-1]:
                    self.distance_matrix[i][j] = self.distance_matrix[i-1][j-1]
                    self.backtrace_matrix[i][j] = self.NO_EDIT
                else:
                    # Calculate operation costs
                    minimum = min(
                        self.distance_matrix[i][j-1],  # Insertion
                        self.distance_matrix[i-1][j],  # Deletion
                        self.distance_matrix[i-1][j-1]  # Replacement
                    )
            
                    if minimum == self.distance_matrix[i][j-1]: 
                        self.distance_matrix[i][j] = minimum + self.insert_cost
                        self.backtrace_matrix[i][j] = self.INSERT
                    elif minimum == self.distance_matrix[i-1][j]: 
                        self.distance_matrix[i][j] = minimum + self.delete_cost
                        self.backtrace_matrix[i][j] = self.DELETE
                    else: 
                        self.distance_matrix[i][j] = minimum + self.replace_cost
                        self.backtrace_matrix[i][j] = self.REPLACE

    def get_edit_distance(self):
        """
        Returns the Levenshtein distance (edit distance) between the source and target strings.

        Returns:
            int: The minimum cost to transform the source string into the target string.
        """
        return self.distance_matrix[self.M][self.N]
    
    def backtrace_to_ascii(self):
        """
        Constructs a visual ASCII representation of the backtrace matrix.

        Returns:
            np.ndarray: A matrix with ASCII symbols representing the operations:
                        'N' for no operation,
                        'I' for insertion,
                        'D' for deletion,
                        'R' for replacement.
        """
        ascii_backtrace = np.array([["N" for _ in range((self.N)+1)] for _ in range((self.M)+1)])
        
        for i in range(1, (self.M)+1):
            for j in range(1, (self.N)+1):
                if self.backtrace_matrix[i][j] == self.INSERT:
                    ascii_backtrace[i][j] = 'I'
                elif self.backtrace_matrix[i][j] == self.DELETE:
                    ascii_backtrace[i][j] = 'D'
                elif self.backtrace_matrix[i][j] == self.REPLACE:
                    ascii_backtrace[i][j] = 'R'
                elif self.backtrace_matrix[i][j] == self.NO_EDIT:
                    ascii_backtrace[i][j] = 'N'

        for i in range((self.M)+1):
            ascii_backtrace[i][0] = self.source[i-1] if i > 0 else '#'
        for j in range((self.N)+1):
            ascii_backtrace[0][j] = self.target[j-1] if j > 0 else '#'

        return ascii_backtrace

    def operations_history(self):
        """
        Traces back the operations performed to transform the source string into the target string.

        Returns:
            list: A list of dictionaries with detailed operation information:
              - 'operation': The type of operation ('inserting', 'deleting', 'replacing', 'no_edit').
              - 'char_index_source': The index in the source string (None for insert).
              - 'char_index_target': The index in the target string (None for delete).
              - 'source_char': The character from the source string (None for insert).
              - 'target_char': The character from the target string (None for delete).
        """
        operations = []

        i, j = self.M, self.N
        while i > 0 or j > 0:
            current_operation = self.backtrace_matrix[i][j]
            if current_operation == self.INSERT:  # INSERT
                operations.append({"operation": 'inserting', 
                                   "char_index" : j, 
                                   "source_char" : None,
                                   "target_char" : (j, self.target[j-1])})
                j -= 1
            elif current_operation == self.DELETE:  # DELETE
                operations.append({"operation": 'deleting', 
                                   "char_index" : i, 
                                   "source_char" : (i, self.source[i-1]),
                                   "target_char" : None})
                i -= 1
            elif current_operation == self.REPLACE:  # REPLACE
                operations.append({"operation": 'replacing', 
                                   "char_index" : i, 
                                   "source_char" : (i, self.source[i-1]),
                                   "target_char" : (j, self.target[j-1])})
                i -= 1
                j -= 1
            else:
                operations.append({"operation": 'no_edit', 
                                   "char_index" : i, 
                                   "source_char" : (i, self.source[i-1]),
                                   "target_char" : (j, self.target[j-1])})
                i -= 1
                j -= 1
        
        return operations[::-1]


    # Eenumerator for operations
    @property 
    def NO_EDIT(self): return 0
    @property 
    def INSERT(self): return 1
    @property
    def DELETE(self): return 2
    @property 
    def REPLACE(self): return 3

