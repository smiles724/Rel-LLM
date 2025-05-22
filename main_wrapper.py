import subprocess

# only classification tasks
datasets_with_tasks = {'rel-event': ['user-ignore', 'user-repeat'], 'rel-amazon': ['user-churn', 'item-churn'], 'rel-hm': ['user-churn'],
                       'rel-stack': ['user-engagement', 'user-badge'], 'rel-trial': ['study-outcome'], 'rel-f1': ['driver-top3'],   # todo: 'driver-dnf'
                       'rel-avito': ['user-visits', 'user-clicks']}


def run_script(dataset, task, display_real_time=True):
    command = ["python", "main.py", "--dataset", dataset, "--task", task, "--dropout", "0.1", "--val_steps", "20", "--val_size", "256"]
    print(f"Running dataset: {dataset}, task: {task}")

    if display_real_time:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)  # Use subprocess.Popen to stream output in real-time
        while True:  # Stream stdout and stderr in real-time
            output = process.stdout.readline()
            error = process.stderr.readline()
            if output == '' and error == '' and process.poll() is not None: break  # Process has finished
            if output: print(output.strip())  # Print stdout in real-time
            if error: print(error.strip())  # Print stderr in real-time
        return_code = process.wait()   # Wait for the process to finish and get the return code
        if return_code == 0:
            print(f"Successfully completed dataset: {dataset}, task: {task}")
        else:
            print(f"Failed to run dataset: {dataset}, task: {task}")
            print(process.stderr.read())  # Print any remaining stderr
    else:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Successfully completed dataset: {dataset}, task: {task}")
        else:
            print(f"Failed to run dataset: {dataset}, task: {task}")
            print(result.stderr)


if __name__ == "__main__":
    # Iterate through all datasets and their tasks
    for dataset_name, tasks in datasets_with_tasks.items():
        for task_name in tasks:
            run_script(dataset_name, task_name)
