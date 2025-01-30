import pandas as pd
import os

# BLEU
import nltk
nltk.download('punkt')
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize


# BERTScore
from bert_score import score

# BARTScore
from bart_score import BARTScorer

# Rouge
from rouge import Rouge

class Evaluation:
    """
    A class for evaluating different metrics on generated evidence.

    Args:
        evaluation_file (str): The path to the evaluation file containing the generated and true evidence.
        metrics_folder (str): The path to the folder where the evaluation metrics will be saved.

    Attributes:
        df_eval (pandas.DataFrame): The DataFrame containing the evaluation data.
        file_name (str): The name of the evaluation file.
        folder_path (str): The path to the metrics folder.
        df_bleu (pandas.DataFrame): The DataFrame for BLEU scores.
        df_BERTscore (pandas.DataFrame): The DataFrame for BERTScore.
        df_rouge (pandas.DataFrame): The DataFrame for ROUGE scores.
        df_BARTscore (pandas.DataFrame): The DataFrame for BARTScore.

    Methods:
        calculate_rouge(row): Calculates the ROUGE scores for a given row of generated and true evidence.
        calculate_BARTscore(row): Calculates the BARTScore for a given row of generated and true evidence.
        calculate_BERTscore(row): Calculates the BERTScore for a given row of generated and true evidence.
        calculate_bleu(row): Calculates the BLEU score for a given row of generated and true evidence.
        evaluate_BARTscore(): Evaluates BARTScore for all rows in the evaluation data and saves the results to a CSV file.
        evaluate_BLEU(): Evaluates BLEU score for all rows in the evaluation data and saves the results to a CSV file.
        evaluate_ROUGE(): Evaluates ROUGE scores for all rows in the evaluation data and saves the results to a CSV file.
        evaluate_BERTscore(): Evaluates BERTScore for all rows in the evaluation data and saves the results to a CSV file.
        evaluate_all_metrics(): Evaluates all metrics (BARTScore, BLEU, ROUGE, BERTScore) and saves the results to CSV files.
        calculate_mean(): Calculates the mean scores for all metrics.

    """

    def __init__(self, evaluation_file, metrics_folder):
        self.df_eval = pd.read_csv(evaluation_file)

        self.file_name = os.path.splitext(os.path.basename(evaluation_file))[0]
        self.folder_path = os.path.dirname(metrics_folder)

        self.df_bleu = self.df_eval.copy()

        self.df_BERTscore = self.df_eval.copy()
        self.df_rouge = self.df_eval.copy()
        self.df_BARTscore = self.df_eval.copy()


    def calculate_rouge(self, row):
        """
        Calculates the ROUGE scores for a given row of generated and true evidence.

        Args:
            row (pandas.Series): The row containing the generated and true evidence.

        Returns:
            pandas.Series: The ROUGE scores (ROUGE-1 F, ROUGE-1 P, ROUGE-1 R, ROUGE-2 F, ROUGE-2 P, ROUGE-2 R, ROUGE-L F, ROUGE-L P, ROUGE-L R).

        """
        rouge = Rouge()
        scores = rouge.get_scores(row['Generated Evidence'], row['True Evidence'])[0]
        return pd.Series({
            'ROUGE-1 F': scores['rouge-1']['f'],
            'ROUGE-1 P': scores['rouge-1']['p'],
            'ROUGE-1 R': scores['rouge-1']['r'],
            'ROUGE-2 F': scores['rouge-2']['f'],
            'ROUGE-2 P': scores['rouge-2']['p'],
            'ROUGE-2 R': scores['rouge-2']['r'],
            'ROUGE-L F': scores['rouge-l']['f'],
            'ROUGE-L P': scores['rouge-l']['p'],
            'ROUGE-L R': scores['rouge-l']['r']
        })

    def calculate_BARTscore(self, row):
        """
        Calculates the BARTScore for a given row of generated and true evidence.

        Args:
            row (pandas.Series): The row containing the generated and true evidence.

        Returns:
            float: The BARTScore.

        """
        bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-mnli')
        return bart_scorer.score([row['True Evidence']], [row['Generated Evidence']], batch_size=4)[0]

    def calculate_BERTscore(self, row):
        """
        Calculates the BERTScore for a given row of generated and true evidence.

        Args:
            row (pandas.Series): The row containing the generated and true evidence.

        Returns:
            pandas.Series: The BERTScore (Precision, Recall, F1 Score).

        """
        P, R, F1 = score([row['Generated Evidence']], [row['True Evidence']], lang='en')
        return pd.Series({'Precision': P.item(), 'Recall': R.item(), 'F1 Score': F1.item()})

    def calculate_bleu(self, row):
        """
        Calculates the BLEU score for a given row of generated and true evidence.

        Args:
            row (pandas.Series): The row containing the generated and true evidence.

        Returns:
            float: The BLEU score.

        """
        true_tokens = word_tokenize(row['True Evidence'].lower())
        generated_tokens = word_tokenize(row['Generated Evidence'].lower())
        return sentence_bleu([true_tokens], generated_tokens)
    
    def evaluate_BARTscore(self):
        """
        Evaluates BARTScore for all rows in the evaluation data and saves the results to a CSV file.

        """
        if not os.path.exists(os.path.join(self.folder_path, f"{self.file_name}_BART.csv")):
            self.df_BARTscore['bartscore'] = self.df_eval.apply(self.calculate_BARTscore, axis=1)
            new_file = os.path.join(self.folder_path, f"{self.file_name}_BART.csv")
            self.df_BARTscore.to_csv(new_file, index=False)

    def evaluate_BLEU(self):
        """
        Evaluates BLEU score for all rows in the evaluation data and saves the results to a CSV file.

        """
        if not os.path.exists(os.path.join(self.folder_path, f"{self.file_name}_BLEU.csv")):
            self.df_bleu['bleu_score'] = self.df_eval.apply(self.calculate_bleu, axis=1)
            new_file = os.path.join(self.folder_path, f"{self.file_name}_BLEU.csv")
            self.df_bleu.to_csv(new_file, index=False)

    def evaluate_ROUGE(self):
        """
        Evaluates ROUGE scores for all rows in the evaluation data and saves the results to a CSV file.

        """
        if not os.path.exists(os.path.join(self.folder_path, f"{self.file_name}_ROUGE.csv")):
            rouge_scores = self.df_eval.apply(self.calculate_rouge, axis=1)
            self.df_rouge = pd.concat([self.df_rouge, rouge_scores], axis=1)
            new_file = os.path.join(self.folder_path, f"{self.file_name}_ROUGE.csv")
            self.df_rouge.to_csv(new_file, index=False)
    
    def evaluate_BERTscore(self):
        """
        Evaluates BERTScore for all rows in the evaluation data and saves the results to a CSV file.

        """
        if not os.path.exists(os.path.join(self.folder_path, f"{self.file_name}_BERT.csv")):
            BERT_scores = self.df_eval.apply(self.calculate_BERTscore, axis=1)
            self.df_BERTscore = pd.concat([self.df_BERTscore, BERT_scores], axis=1)
            new_file = os.path.join(self.folder_path, f"{self.file_name}_BERT.csv")
            self.df_BERTscore.to_csv(new_file, index=False)
       
    def evaluate_all_metrics(self):
        """
        Evaluates all metrics (BARTScore, BLEU, ROUGE, BERTScore) and saves the results to CSV files.

        """
        self.evaluate_BARTscore()
        self.evaluate_BLEU()
        self.evaluate_ROUGE()
        self.evaluate_BERTscore()

    def calculate_mean(self):
        """
        Calculates the mean scores for all metrics.

        Returns:
            pandas.Series: The mean scores (BLEU, BARTScore, F1 BERTScore, ROUGE-1 F, ROUGE-2 F, ROUGE-L F).

        """
        mean_bleu = self.df_bleu['bleu_score'].mean()
        mean_BARTscore = self.df_BARTscore['bartscore'].mean()
        mean_BERTscore = self.df_BERTscore['F1 Score'].mean()
        mean_ROUGE = self.df_rouge[['ROUGE-1 F', 'ROUGE-2 F', 'ROUGE-L F']].mean()
        return pd.Series({'BLEU': mean_bleu, 'BARTScore': mean_BARTscore, 'F1 BERTScore': mean_BERTscore, 'ROUGE-1 F': mean_ROUGE['ROUGE-1 F'], 'ROUGE-2 F': mean_ROUGE['ROUGE-2 F'], 'ROUGE-L F': mean_ROUGE['ROUGE-L F']})


       