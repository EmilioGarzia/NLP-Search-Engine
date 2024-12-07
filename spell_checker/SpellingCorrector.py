# Spelling Corrector
#
# @author Emilio Garzia, 2024

from nltk.corpus import words
from nltk.tokenize import word_tokenize
from collections import Counter
from .Levenshtein import levenshtein

################################################################################################

"""
This is a vocabulary made by reuters corpus (inserted in this module just for demo).
By default, if you don't specify any vocabulary will be upload the "english words vocabulary" (from NLTK too)
"""
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "reuters_vocabulary.txt")
with open(file_path, "r") as file:
    reuters_vocabulary = [word.strip() for word in file]

################################################################################################

class SpellCorrector:
    """
    A class to perform spell correction on a given string using a vocabulary. 
    It uses Levenshtein distance to find candidates and computes probabilities 
    based on word frequency and edit distance.

    Attributes:
        string (str): The input string to be corrected.
        vocabulary (list or set): The list or set of valid words. If not provided, 
            a default English vocabulary is used.
        n_edit (int): Maximum edit distance to consider a word as a candidate.
        misspelled_words (list): A list of dictionaries containing details about 
            misspelled words, their candidates, and probabilities.
    """

    def __init__(self, string, vocabulary=None, n_edit=2):
        """
        Initializes the SpellCorrector class.

        Args:
            string (str): The string to be corrected.
            vocabulary (list or set, optional): The vocabulary of valid words. 
                Defaults to None, in which case the English vocabulary from `nltk` is used.
            n_edit (int): Maximum allowable edit distance for candidates. Defaults to 2.
        """
        self.string = string
        self.vocabulary = vocabulary
        self.n_edit = n_edit
        self.misspelled_words = []

        # Load English vocabulary if no vocabulary is provided
        if self.vocabulary is None:
            self.vocabulary = set(words.words())  # Load default vocabulary from nltk
            self.vocabulary = sorted(self.vocabulary)

        # Main steps
        self.__find_misspelled__()
        self.__find_candidates__()
        self.__compute_probabilities__()
        self.__best_candidate__()

    def retrive_corrected(self):
        """
        Retrieves the corrected string by replacing misspelled words with the best candidates.

        Returns:
            str: The corrected version of the input string.
        """
        corrected_string = " ".join(self.string)  # Reconstruct the input string from tokens

        # Replace misspelled words with their best candidates
        for misspelled_word in self.misspelled_words:
            misspelled = misspelled_word["misspelled"]
            best_candidate = misspelled_word["best_candidates"]
            corrected_string = corrected_string.replace(misspelled, best_candidate)

        return corrected_string

    def __best_candidate__(self):
        """
        Identifies the best candidate for each misspelled word based on the computed probabilities.
        Updates the `best_candidates` field for each word in `self.misspelled_words`.
        """
        for misspelled_word in self.misspelled_words:
            candidates = misspelled_word["candidates"]
            probabilities = misspelled_word["probabilities"]
            best_probability = -1
            best_candidate = ""

            # Find the candidate with the highest probability
            for candidate, probability in zip(candidates, probabilities):
                if probability > best_probability:
                    best_probability = probability
                    best_candidate = candidate

            misspelled_word["best_candidates"] = best_candidate

    def __compute_probabilities__(self):
        """
        Computes the probabilities for each candidate based on edit distance 
        and frequency in the vocabulary. Updates the `probabilities` field 
        for each word in `self.misspelled_words`.
        """
        vocabulary_counter = Counter(self.vocabulary)  # Frequency of words in the vocabulary

        for misspelled_word in self.misspelled_words:
            candidates = misspelled_word["candidates"]
            misspelled = misspelled_word["misspelled"]
            misspelled_word["probabilities"] = []  # Initialize probabilities list

            for candidate in candidates:
                n_edit = levenshtein(misspelled, candidate, replace_cost=self.n_edit).get_edit_distance()
                candidate_frequency = vocabulary_counter[candidate]
                                
                # Probability
                distance_term = 1 / (n_edit + 1)
                lambda_factor = 0.5  # distance and frequency balancing
                probability = (candidate_frequency * distance_term) ** lambda_factor
                
                misspelled_word["probabilities"].append(probability)

    def __find_candidates__(self):
        """
        Finds candidate words for each misspelled word based on the edit distance.
        Updates the `candidates` field for each word in `self.misspelled_words`.
        """
        for misspelled in self.misspelled_words:
            for word in self.vocabulary:
                if levenshtein(misspelled["misspelled"], word, replace_cost=2).get_edit_distance() <= self.n_edit:
                    misspelled["candidates"].append(word)

    def __find_misspelled__(self):
        """
        Tokenizes the input string and identifies words not found in the vocabulary.
        Populates the `misspelled_words` list with details about each misspelled word.
        """
        self.string = word_tokenize(self.string)  
        self.string = [word.lower() for word in self.string]  

        for word in self.string:
            if word.isalpha():  # Ignore numbers and special characters
                if word not in self.vocabulary:
                    self.misspelled_words.append({
                        "misspelled": word,
                        "candidates": [],
                        "probabilities": [],
                    })