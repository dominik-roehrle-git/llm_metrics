from StoppingCriteria import TokenStoppingCriteria
import transformers
import re
import pandas as pd
import pandas as pd
import re


class GenerateEvidence:
    """
    A class for generating evidence based on given prompts using a pre-trained language model.

    Args:
        model (object): The pre-trained language model.
        file_example_path (str): The file path of the example data.
        file_test_path (str): The file path of the test data.
        number_examples (int): The number of examples to generate.
        max_tokens (int): The maximum number of tokens to generate in the output.

    Attributes:
        number_examples (int): The number of examples to generate.
        model (object): The pre-trained language model.
        generate_text (object): The text generation pipeline.
        example_data (DataFrame): The example data.
        hops_2_supports_example (DataFrame): The example data with 2 hops and 'SUPPORTS' label.
        hops_3_supports_example (DataFrame): The example data with 3 hops and 'SUPPORTS' label.
        hops_4_supports_example (DataFrame): The example data with 4 hops and 'SUPPORTS' label.
        hops_2_refutes_example (DataFrame): The example data with 2 hops and 'REFUTES' label.
        hops_3_refutes_example (DataFrame): The example data with 3 hops and 'REFUTES' label.
        hops_4_refutes_example (DataFrame): The example data with 4 hops and 'REFUTES' label.
        test_data (DataFrame): The test data.
        hops_2_supports_test (DataFrame): The test data with 2 hops and 'SUPPORTS' label.
        hops_3_supports_test (DataFrame): The test data with 3 hops and 'SUPPORTS' label.
        hops_4_supports_test (DataFrame): The test data with 4 hops and 'SUPPORTS' label.
        hops_2_refutes_test (DataFrame): The test data with 2 hops and 'REFUTES' label.
        hops_3_refutes_test (DataFrame): The test data with 3 hops and 'REFUTES' label.
        hops_4_refutes_test (DataFrame): The test data with 4 hops and 'REFUTES' label.
        prompt_df (DataFrame): The dataframe to store the prompt data.
        generated_df (DataFrame): The dataframe to store the generated evidence data.

    Methods:
        reset_data(): Resets the dataframes to their initial state.
        set_generate_text(max_tokens, number_examples): Sets up the text generation pipeline.
        select_from_df(supports, hops, ending): Selects a row from the dataframe based on the given supports, hops, and ending.
        clean_evidence(evidence): Cleans the evidence by removing unnecessary characters and extracting topics.
        generate_prompt_option(supports_option, hops_option, old_hops, old_supports): Generates the new supports and hops options for the prompt.
        get_prompt(supports_option, hops_option): Generates the prompt based on the supports and hops options.
        generate_evidence(prompt, supports, hops, number_generations): Generates evidence based on the given prompt, supports, hops, and number of generations.
    """

    def __init__(self, model, file_example_path, file_test_path, number_examples, max_tokens):
        self.number_examples = number_examples
        self.model = model
        
        self.generate_text = self.set_generate_text(max_tokens, number_examples)

        self.example_data = pd.read_json(file_example_path)
        self.hops_2_supports_example = self.example_data.loc[(self.example_data['hops'] == 2) & (self.example_data['label'] == 'SUPPORTS')]
        self.hops_3_supports_example = self.example_data.loc[(self.example_data['hops'] == 3) & (self.example_data['label'] == 'SUPPORTS')]
        self.hops_4_supports_example = self.example_data.loc[(self.example_data['hops'] == 4) & (self.example_data['label'] == 'SUPPORTS')]

        self.hops_2_refutes_example = self.example_data.loc[(self.example_data['hops'] == 2) & (self.example_data['label'] == 'REFUTES')]
        self.hops_3_refutes_example = self.example_data.loc[(self.example_data['hops'] == 3) & (self.example_data['label'] == 'REFUTES')]
        self.hops_4_refutes_example = self.example_data.loc[(self.example_data['hops'] == 4) & (self.example_data['label'] == 'REFUTES')]

        self.test_data = pd.read_json(file_test_path)
        self.hops_2_supports_test = self.test_data.loc[(self.test_data['hops'] == 2) & (self.test_data['label'] == 'SUPPORTS')]
        self.hops_3_supports_test = self.test_data.loc[(self.test_data['hops'] == 3) & (self.test_data['label'] == 'SUPPORTS')]
        self.hops_4_supports_test = self.test_data.loc[(self.test_data['hops'] == 4) & (self.test_data['label'] == 'SUPPORTS')]

        self.hops_2_refutes_test = self.test_data.loc[(self.test_data['hops'] == 2) & (self.test_data['label'] == 'REFUTES')]
        self.hops_3_refutes_test = self.test_data.loc[(self.test_data['hops'] == 3) & (self.test_data['label'] == 'REFUTES')]
        self.hops_4_refutes_test = self.test_data.loc[(self.test_data['hops'] == 4) & (self.test_data['label'] == 'REFUTES')]


        self.prompt_df = pd.DataFrame(columns=['claim', 'label', 'evidence', 'hops'])
        self.generated_df = pd.DataFrame(columns=['True Evidence', 'Generated Evidence', 'Hops', 'Label'])

    def reset_data(self):
        """
        Resets the dataframes to their initial state.
        """
        self.hops_2_supports_example = self.example_data.loc[(self.example_data['hops'] == 2) & (self.example_data['label'] == 'SUPPORTS')]
        self.hops_3_supports_example = self.example_data.loc[(self.example_data['hops'] == 3) & (self.example_data['label'] == 'SUPPORTS')]
        self.hops_4_supports_example = self.example_data.loc[(self.example_data['hops'] == 4) & (self.example_data['label'] == 'SUPPORTS')]

        self.hops_2_refutes_example = self.example_data.loc[(self.example_data['hops'] == 2) & (self.example_data['label'] == 'REFUTES')]
        self.hops_3_refutes_example = self.example_data.loc[(self.example_data['hops'] == 3) & (self.example_data['label'] == 'REFUTES')]
        self.hops_4_refutes_example = self.example_data.loc[(self.example_data['hops'] == 4) & (self.example_data['label'] == 'REFUTES')]

        self.hops_2_supports_test = self.test_data.loc[(self.test_data['hops'] == 2) & (self.test_data['label'] == 'SUPPORTS')]
        self.hops_3_supports_test = self.test_data.loc[(self.test_data['hops'] == 3) & (self.test_data['label'] == 'SUPPORTS')]
        self.hops_4_supports_test = self.test_data.loc[(self.test_data['hops'] == 4) & (self.test_data['label'] == 'SUPPORTS')]

        self.hops_2_refutes_test = self.test_data.loc[(self.test_data['hops'] == 2) & (self.test_data['label'] == 'REFUTES')]
        self.hops_3_refutes_test = self.test_data.loc[(self.test_data['hops'] == 3) & (self.test_data['label'] == 'REFUTES')]
        self.hops_4_refutes_test = self.test_data.loc[(self.test_data['hops'] == 4) & (self.test_data['label'] == 'REFUTES')]

        self.prompt_df = pd.DataFrame(columns=['claim', 'label', 'evidence', 'hops'])
        self.generated_df = pd.DataFrame(columns=['True Evidence', 'Generated Evidence', 'Hops', 'Label', 'Topics'])

    def set_generate_text(self, max_tokens, number_examples):
        """
        Sets up the text generation pipeline.

        Args:
            max_tokens (int): The maximum number of tokens to generate in the output.
            number_examples (int): The number of examples to generate.

        Returns:
            object: The text generation pipeline.
        """
        self.number_examples = number_examples
        sentinel_token_ids = self.model.tokenizer("###", add_special_tokens=False, return_tensors="pt").input_ids.to("cuda")
        self.stopping_criteria_list = transformers.StoppingCriteriaList([
            TokenStoppingCriteria(sentinel_token_ids=sentinel_token_ids, starting_idx=0, counter=0, stop_counter=number_examples)
        ])
        self.generate_text = transformers.pipeline(
            model=self.model.model_causal_lm, 
            tokenizer=self.model.tokenizer,
            return_full_text=False,  
            task='text-generation',
            stopping_criteria=self.stopping_criteria_list,  # without this the model continues to invent new claims and evidences
            max_new_tokens=max_tokens,  # max number of tokens to generate in the output
        )
        return self.generate_text

    def select_from_df(self, supports, hops, ending):
        """
        Selects a row from the dataframe based on the given supports, hops, and ending.

        Args:
            supports (str): The supports option ('SUPPORTS' or 'REFUTES').
            hops (int): The number of hops.
            ending (str): The ending of the dataframe name ('example' or 'test').

        Returns:
            DataFrame: The selected row from the dataframe.
        """
        df_name = f"hops_{hops}_{supports.lower()}_{ending}"
        df = getattr(self, df_name)
        selected_row = df.iloc[0:1]
        if df.empty:
            raise Exception("Not enough samples in the dataframe. Please choose a lower number of examples/generations")
        df = df.iloc[1:]
        setattr(self, df_name, df)
        return selected_row

    def clean_evidence(self, evidence):
        """
        Cleans the evidence by removing unnecessary characters and extracting topics/reference wikipedia names.

        Args:
            evidence (str): The evidence to clean.

        Returns:
            tuple: A tuple containing the cleaned evidence and the extracted topics.
        """
        evidence = " ".join(evidence)
        pattern = r'\[([^\]]+)\]'
        topics = re.findall(pattern, evidence)
        topics = list(set(topics))
        topics = ", ".join(topics)
        evidence = evidence.strip()
        return evidence, topics
    
    def generate_prompt_option(self, supports_option, hops_option, old_hops, old_supports):
        """
        Generates the new supports and hops pairs for the prompt.

        Args:
            supports_option (str): The supports option ('SUPPORTS', 'REFUTES', or 'SUPPORTS+REFUTES').
            hops_option (str): The hops option ('2', '3', '4', or 'mixed').
            old_hops (int): The previous number of hops.
            old_supports (str): The previous supports option.

        Returns:
            tuple: A tuple containing the new supports and hops specification.
        """
        if supports_option == "SUPPORTS+REFUTES":
            if old_supports == "SUPPORTS":
                new_supports = "REFUTES"
            else:
                new_supports = "SUPPORTS"
        else:
            new_supports = supports_option

        if hops_option == "mixed":
            if old_hops == 2:
                new_hops = 3
            elif old_hops == 3:
                new_hops = 4
            else:
                new_hops = 2
        else:
            new_hops = hops_option
        return new_supports, new_hops

    def get_prompt(self, supports_option, hops_option):
        """
        Generates the prompt based on the supports and hops options.

        Args:
            supports_option (str): The supports option ('SUPPORTS', 'REFUTES', or 'SUPPORTS+REFUTES').
            hops_option (str): The hops option ('2', '3', '4', or 'mixed').

        Returns:
            str: The generated prompt.
        """
        if supports_option == "SUPPORTS+REFUTES":
            old_supports = "REFUTES"
        else:
            old_supports = supports_option

        if hops_option == "mixed": 
            old_hops = 4
        else:
            old_hops = hops_option

        for example_index in range(self.number_examples):
            new_supports, new_hops = self.generate_prompt_option(supports_option, hops_option, old_hops, old_supports)
            selected_row = self.select_from_df(new_supports, new_hops, ending="example")
            old_hops, old_supports = new_hops, new_supports
            self.prompt_df = pd.concat([self.prompt_df, selected_row], ignore_index=True)
        prompt = ""
        for index, entry in self.prompt_df.iterrows():
            clean_evidence, _ = self.clean_evidence(entry['evidence'])
            claim = entry['claim']
            prompt += 'CLAIM: ' + str(claim) +  ' EVIDENCE: ' + str(clean_evidence).replace('"', '').replace("'", '') + " ### " 
        return prompt

    def generate_evidence(self, prompt, supports, hops, number_generations=None):
        """
        Generates evidence based on the given prompt, supports, hops, and number of generations.

        Args:
            prompt (str): The prompt for generating evidence.
            supports (str): The supports option ('SUPPORTS', 'REFUTES', or 'SUPPORTS+REFUTES').
            hops (int): The number of hops.
            number_generations (int, optional): The number of evidence generations. Defaults to None.

        Raises:
            Exception: If there are not enough samples in the dataframe.

        Returns:
            None
        """
        if number_generations and number_generations > len(getattr(self, f"hops_{hops}_{supports.lower()}_test")):
            raise Exception("Not enough samples in the dataframe. Please choose a lower number of generations")
        
        if number_generations is None:
            number_generations = len(getattr(self, f"hops_{hops}_{supports.lower()}_test"))

        for generation_index in range(number_generations):
            selected_row = self.select_from_df(supports, hops, "test")
            clean_evidence, topics = self.clean_evidence(selected_row['evidence'].item())

            prompt_with_new_claim = prompt + 'CLAIM: ' + str(selected_row['claim'].item()) +  ' EVIDENCE: '

            generated_evidence = self.generate_text(prompt_with_new_claim)
            generated_evidence = str(generated_evidence[0]["generated_text"]).replace('###', '')
            generated_evidence = re.sub(' +', ' ', generated_evidence)

            generated_row = pd.DataFrame({'True Evidence': [clean_evidence], 'Generated Evidence': [generated_evidence.strip()], 
                                          "Hops": [selected_row['hops'].item()], "Label": [selected_row['label'].item()], "Topics": [topics]})
            self.generated_df = pd.concat([self.generated_df, generated_row], ignore_index=True)
        return self.generated_df



    