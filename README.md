# Factuality Evaluation of Llama2-7b: Insights from the HoVER Dataset (Text Generation and Metrics: BARTScore, BERTScore, BLUE)

This repository provides the Text Generation for Llama2-7b-chat and the Evaluation Metrics BARTScore, BERTScore, BLUE.
The Evaluation for FActScore is here: 

## Installation (Python 3.11 and Windows)

- Create an account on https://huggingface.co/ and follow the instructions: https://huggingface.co/meta-llama and request request access to Llama: https://llama.meta.com/llama-downloads
- create a virtual environment
- install pytorch: pip3 install torch --index-url https://download.pytorch.org/whl/cu121
- install the packages from requirements.txt: pip install -r requirements.txt
- I had several problems to install bitsandbytes and it seems to not be supported for windows (also bitsandbytes-windows didn't work on my machine)
- I solved the problem by running this command (reference: https://github.com/d8ahazard/sd_dreambooth_extension/issues/7): pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.1-py3-none-win_amd64.whl 
- Please run the cell: model = LLama2(model_id="meta-llama/Llama-2-7b-chat-hf") only one time. Otherwise restart the kernel in generate.ipynb 

## Run the code
- to see the results for the HoVER dataset for Llama2-7b you can directly go to evaluate.ipynb
- to produce new text generations go to generate.ipynb


