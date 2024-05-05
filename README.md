# Ollama Bench

Ollama Bench is a benchmarking program that I (meaning me, ChatGPT and Claude) developed based on the [Ollama Python Library](https://github.com/ollama/ollama-python) to test the performance and capabilities of the Parallel requests and Multi Model capabilities released in [Ollama 0.1.33](https://github.com/ollama/ollama/releases/tag/v0.1.33).

---

## Requirements

- Mac/Linux
- Python3
- Ollama >=0.1.33

---

## Usage

1. Clone this repo `git clone https://github.com/llamanator-project/ollama-bench.git` and `cd ollama-bench`
2. Add a bank of prompts as a `somefile.txt` format with each prompt on a new line separated by a comma. Refer to the files in the `./sample-prompts` dir for the format, or just use one of the samples for your run
3. Install the required python packages `pip install -r requirements.txt`
4. Run `python3 ollama-bench.py -f ./your-question-file.txt` and you will be prompted for some input
5. Fill in the inputs and confirm to start the run
    - `Enter Ollama host URL:` Enter the target of your Ollama server like http://127.0.0.1:11434
    - `Available models: {model list} Select up to 10 models by number (e.g., 1, 3, 5):` The list of available models on your server will be queried and presented. Select the models you want to use like `1, 3, 5` etc. This will test the multiple model loading capabilities of Ollama 0.1.33. The selected models will be used in a round-robin method to query throughout the run
    - `How many concurrent requests would you like to run?` Select how many times you want to prompt the models. If your prompt list is shorter than the number you put here, the program will loop over your prompt list to complete the number requested
    - `How much time in seconds should wait between requests?` This is how long to wait in between prompts. Lowest is 1 second.
    - `Enter the timeout duration in seconds (after which a request will be marked as running for too long):` Choose a value in seconds for how long before a request is automatically stopped. This comes in handy as some models start looping and generating nonsense forever. The program will monitor how long the run takes and automatically kill the request after the defined time.
    - `Please confirm the details of the run by typing 'yes':` You will be prompted to confirm the details of the run by typing `yes`
6. View the results in the `./ollama-benchmark-results/{current_datetime}-benchmark` dir for your specific run

---

## Outputs

- During the run, all successful requests and errors will be put into individual `.csv` files in the `./ollama-benchmark-results` dir under a dir called `{current_datetime}-benchmark` for each run.
- At the end of the run, or if you hit `ctrl + c` one time, the results will be compiled into `benchmark_results.csv` and `error_log.csv`
  - The `benchmark_results.csv` contains all of the prompts, answers, model and stats about each request. At the bottom of the csv, the results are summarized with the following:
    - `Total Tokens Received in Response:` How many tokens were received across all requests
    - `Highest Tokens per Second:` Faster tokens per second response time
    - `Lowest Tokens per Second:` Slowest tokens per second response time
    - `Averages:` Average tokens per second across all requests
    - `Model Usage Summary:` The number of times each model selected for this run was used

---

## Ollama Setup

In order to full test this, you will need Ollama 0.1.33 or newer and make sure you have added the following options on your appropriate OS:

- `OLLAMA_NUM_PARALLEL=x`: This sets the number of parallel requests Ollama will accept. An error will be sent when you exceed the number set here.
- `OLLAMA_MAX_LOADED_MODELS=x`: This set the number of models that Ollama will be allowed to load at one time. 

You will need to set these and tune them to determine the best results for your particular system.

---

## Issues and Improvements

There will likely be issues. I am not a developer and I fully leveraged ChatGPT and Claude to write this.

If you find an issue, please feel free to open an issue on the Issues tab or submit a PR for the fix.

If you want to improve the script feel free to submit a PR and I will review and merge.