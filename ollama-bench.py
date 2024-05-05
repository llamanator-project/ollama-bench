import argparse
import threading
import time
from datetime import datetime
import csv
import os
from pydantic import BaseModel, Field, validator
import ollama

class Message(BaseModel):
    role: str
    content: str

class OllamaResponse(BaseModel):
    model: str
    created_at: datetime
    message: Message
    done: bool
    total_duration: int
    load_duration: int = 0
    prompt_eval_count: int = Field(-1, validate_default=True)
    prompt_eval_duration: int
    eval_count: int
    eval_duration: int

    @validator("prompt_eval_count")
    @classmethod
    def validate_prompt_eval_count(cls, value):
        if value == -1:
            print("\nWarning: prompt token count was not provided, potentially due to prompt caching.")
            return 0  # Set default value
        return value

def nanosec_to_sec(nanosec):
    return nanosec / 1_000_000_000

model_usage = {}  # Model usage counter

def run_benchmark(model_name: str, prompt: str, verbose: bool, directory: str, index: int, timeout: int):
    print(f"Request {index + 1}: Starting request for: '{prompt}' using model {model_name} at {datetime.now()}")

    try:
        client = ollama.Client(host=os.getenv('OLLAMA_HOST', 'http://127.0.0.1:11434'))
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        stream = client.chat(model=model_name, messages=messages, stream=True)

        response_text = ""
        stats_collected = False
        start_time = time.time()
        last_log_time = start_time

        for chunk in stream:
            if 'message' in chunk:
                message_content = chunk['message']['content']
                response_text += message_content
                if verbose:
                    print(f"Received chunk: {message_content}")

            if not stats_collected and 'done' in chunk and chunk['done']:
                model_response = OllamaResponse.parse_obj(chunk)
                stats = get_stats_dict(model_response)
                response_text += "\n\n" + "\n".join([f"{k}: {v}" for k, v in stats.items()])
                if verbose:
                    print("\n".join([f"{k}: {v}" for k, v in stats.items()]))
                stats_collected = True
                model_usage[model_name] = model_usage.get(model_name, 0) + 1

            current_time = time.time()
            if current_time - last_log_time >= 30:
                elapsed_time = current_time - start_time
                print(f"Request {index + 1}: Elapsed time: {elapsed_time:.2f} seconds")
                last_log_time = current_time

            if current_time - start_time >= timeout:
                raise TimeoutError(f"Request {index + 1}: Timeout reached. Request running for too long.")

        if stats_collected:
            response_data = {'Prompt': prompt, 'Response': response_text, 'Model': model_name, **stats}
            write_to_csv(directory, response_data, index, False)
        else:
            error_message = f"No stats collected for prompt: '{prompt}' using model {model_name}"
            print(f"Request {index + 1}: {error_message}")
            error_data = {'Prompt': prompt, 'Error': error_message, 'Model': model_name}
            write_to_csv(directory, error_data, index, True)

    except TimeoutError as te:
        error_message = str(te)
        print(error_message)
        error_data = {'Prompt': prompt, 'Error': error_message, 'Model': model_name}
        write_to_csv(directory, error_data, index, True)

    except Exception as e:
        error_message = f"Error occurred for prompt: '{prompt}' using model {model_name}. Error: {str(e)}"
        print(f"Request {index + 1}: {error_message}")
        error_data = {'Prompt': prompt, 'Error': error_message, 'Model': model_name}
        write_to_csv(directory, error_data, index, True)

def get_stats_dict(model_response: OllamaResponse) -> dict:
    prompt_ts = model_response.prompt_eval_count / nanosec_to_sec(model_response.prompt_eval_duration) if model_response.prompt_eval_duration > 0 else 0
    response_ts = model_response.eval_count / nanosec_to_sec(model_response.eval_duration) if model_response.eval_duration > 0 else 0
    total_ts = (model_response.prompt_eval_count + model_response.eval_count) / nanosec_to_sec(model_response.total_duration) if model_response.total_duration > 0 else 0
    return {
        'Prompt Eval (t/s)': f"{prompt_ts:.2f}",
        'Response Eval (t/s)': f"{response_ts:.2f}",
        'Total Eval (t/s)': f"{total_ts:.2f}",
        'Eval Count': model_response.eval_count
    }

def write_to_csv(directory: str, data: dict, index: int, is_error: bool):
    prefix = "error_" if is_error else "benchmark_results_"
    csv_path = os.path.join(directory, f'{prefix}{index + 1}.csv')
    file_exists = os.path.isfile(csv_path)

    fieldnames = ['Prompt', 'Response', 'Model', 'Prompt Eval (t/s)', 'Response Eval (t/s)', 'Total Eval (t/s)', 'Eval Count']
    if is_error:
        fieldnames = ['Prompt', 'Error', 'Model']

    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(data)

def merge_csv_files(directory: str):
    csv_files = [file for file in os.listdir(directory) if file.startswith('benchmark_results_') and file.endswith('.csv')]
    merged_csv_path = os.path.join(directory, 'benchmark_results.csv')

    prompt_eval_sum = 0
    response_eval_sum = 0
    total_eval_sum = 0
    eval_count_sum = 0
    row_count = 0

    prompt_eval_min = float('inf')
    prompt_eval_max = 0
    response_eval_min = float('inf')
    response_eval_max = 0
    total_eval_min = float('inf')
    total_eval_max = 0

    with open(merged_csv_path, 'w', newline='') as merged_file:
        fieldnames = ['Prompt', 'Response', 'Model', 'Prompt Eval (t/s)', 'Response Eval (t/s)', 'Total Eval (t/s)', 'Eval Count']
        writer = csv.DictWriter(merged_file, fieldnames=fieldnames)
        writer.writeheader()

        for csv_file in csv_files:
            csv_path = os.path.join(directory, csv_file)
            with open(csv_path, 'r', newline='') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    writer.writerow(row)
                    prompt_eval = float(row['Prompt Eval (t/s)'])
                    response_eval = float(row['Response Eval (t/s)'])
                    total_eval = float(row['Total Eval (t/s)'])
                    eval_count = int(row['Eval Count'])

                    prompt_eval_sum += prompt_eval
                    response_eval_sum += response_eval
                    total_eval_sum += total_eval
                    eval_count_sum += eval_count
                    row_count += 1

                    prompt_eval_min = min(prompt_eval_min, prompt_eval)
                    prompt_eval_max = max(prompt_eval_max, prompt_eval)
                    response_eval_min = min(response_eval_min, response_eval)
                    response_eval_max = max(response_eval_max, response_eval)
                    total_eval_min = min(total_eval_min, total_eval)
                    total_eval_max = max(total_eval_max, total_eval)

            os.remove(csv_path)  # Remove the individual CSV file after merging

    # Calculate averages
    prompt_eval_avg = prompt_eval_sum / row_count if row_count > 0 else 0
    response_eval_avg = response_eval_sum / row_count if row_count > 0 else 0
    total_eval_avg = total_eval_sum / row_count if row_count > 0 else 0

    # Write averages, highest, lowest, and total values to the CSV file
    with open(merged_csv_path, 'a', newline='') as merged_file:
        writer = csv.writer(merged_file)
        writer.writerow([])
        writer.writerow(['Averages:'])
        writer.writerow(['', '', '', f"{prompt_eval_avg:.2f}", f"{response_eval_avg:.2f}", f"{total_eval_avg:.2f}", ''])
        writer.writerow([])
        writer.writerow(['Highest Tokens per Second:'])
        writer.writerow(['', '', '', f"{prompt_eval_max:.2f}", f"{response_eval_max:.2f}", f"{total_eval_max:.2f}", ''])
        writer.writerow([])
        writer.writerow(['Lowest Tokens per Second:'])
        writer.writerow(['', '', '', f"{prompt_eval_min:.2f}", f"{response_eval_min:.2f}", f"{total_eval_min:.2f}", ''])
        writer.writerow([])
        writer.writerow(['Total Tokens Received in Response:'])
        writer.writerow(['', '', '', '', '', '', eval_count_sum])

    # Calculate averages
    prompt_eval_avg = prompt_eval_sum / row_count if row_count > 0 else 0
    response_eval_avg = response_eval_sum / row_count if row_count > 0 else 0
    total_eval_avg = total_eval_sum / row_count if row_count > 0 else 0

    # Write averages, highest, and lowest values to the CSV file
    with open(merged_csv_path, 'a', newline='') as merged_file:
        writer = csv.writer(merged_file)
        writer.writerow([])
        writer.writerow(['Averages:'])
        writer.writerow(['', '', '', f"{prompt_eval_avg:.2f}", f"{response_eval_avg:.2f}", f"{total_eval_avg:.2f}"])
        writer.writerow([])
        writer.writerow(['Highest Tokens per Second:'])
        writer.writerow(['', '', '', f"{prompt_eval_max:.2f}", f"{response_eval_max:.2f}", f"{total_eval_max:.2f}"])
        writer.writerow([])
        writer.writerow(['Lowest Tokens per Second:'])
        writer.writerow(['', '', '', f"{prompt_eval_min:.2f}", f"{response_eval_min:.2f}", f"{total_eval_min:.2f}"])


def merge_error_files(directory: str):
    error_files = [file for file in os.listdir(directory) if file.startswith('error_') and file.endswith('.csv')]
    merged_error_path = os.path.join(directory, 'error_log.csv')

    with open(merged_error_path, 'w', newline='') as merged_file:
        fieldnames = ['Prompt', 'Error', 'Model']
        writer = csv.DictWriter(merged_file, fieldnames=fieldnames)
        writer.writeheader()

        for error_file in error_files:
            error_path = os.path.join(directory, error_file)
            with open(error_path, 'r', newline='') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    writer.writerow(row)

            os.remove(error_path)  # Remove the individual error file after merging

def main():
    host = os.getenv('OLLAMA_HOST')
    if not host:
        host = input("Enter Ollama host URL (e.g., http://147.185.40.120:20021): ")

    parser = argparse.ArgumentParser(description="Run benchmarks on your Ollama models.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity")
    parser.add_argument("-f", "--file", required=True, help="File path to read prompts from.")
    args = parser.parse_args()

    client = ollama.Client(host=host)
    models_data = client.list()
    available_models = [model['name'] for model in models_data['models']]
    print("Available models:")
    for i, model in enumerate(available_models, start=1):
        print(f"{i}. {model}")

    selections = input("Select up to 10 models by number (e.g., 1, 3, 5): ").split(',')
    selected_models = [available_models[int(x.strip()) - 1] for x in selections if x.strip().isdigit() and int(x.strip()) <= len(available_models)]

    with open(args.file, 'r') as file:
        prompts = [line.strip() for line in file if line.strip()]

    num_requests = int(input("How many concurrent requests would you like to run? "))
    delay = float(input("How much time in seconds should wait between requests? "))
    timeout = int(input("Enter the timeout duration in seconds (after which a request will be marked as running for too long): "))

    print (f"\nTarget Ollama Server: {host}")
    print(f"You have selected the following models:")
    for model in selected_models:
        print(model)
    print(f"Number of concurrent requests: {num_requests}")
    print(f"Delay between requests: {delay} seconds")
    print(f"Timeout duration: {timeout} seconds")

    confirmation = input("\nPlease confirm the details of the run by typing 'yes': ")
    if confirmation.lower() != 'yes':
        print("Run canceled.")
        return

    current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    directory = f'./ollama-benchmark-results/{current_datetime}-benchmark'
    os.makedirs(directory, exist_ok=True)

    threads = []
    try:
        prompt_index = 0
        for i in range(num_requests):
            current_model = selected_models[i % len(selected_models)]
            prompt = prompts[prompt_index]
            prompt_index = (prompt_index + 1) % len(prompts)
            thread = threading.Thread(target=run_benchmark, args=(current_model, prompt, args.verbose, directory, i, timeout))
            thread.start()
            threads.append(thread)
            time.sleep(delay)

        # Wait for all threads to finish
        for thread in threads:
            thread.join()

    except KeyboardInterrupt:
        print("Keyboard interrupt received. Waiting for threads to finish...")
        for thread in threads:
            thread.join()
        print("All threads finished.")

    # Merge individual CSV files into a single file
    merge_csv_files(directory)

    # Merge individual error files into a single file
    merge_error_files(directory)

    # Summarize model usage
    with open(os.path.join(directory, 'benchmark_results.csv'), 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Prompt', 'Response', 'Model', 'Prompt Eval (t/s)', 'Response Eval (t/s)', 'Total Eval (t/s)'])
        writer.writerow({})
        writer.writerow({'Prompt': 'Model Usage Summary'})
        for model, count in model_usage.items():
            writer.writerow({'Model': model, 'Prompt': f'Used {count} times'})

if __name__ == "__main__":
    main()