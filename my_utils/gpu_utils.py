import subprocess, re
import os
import time


# Nvidia-smi GPU memory parsing.
# Tested on nvidia-smi 370.23
# TF1.15


def run_command(cmd):
    """Run command, return output as string."""
    output = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    return output.decode("ascii")

def run_async_command(cmd):
    """ Run a command asynchronously"""
    #output = subprocess.Popen(cmd, shell=True)
    #res = os.spawnl(os.P_NOWAIT, cmd)
    res = os.system("nohup " + cmd + " &")
    return "command ran: " + cmd + " res is: " + str(res)
    

def list_available_gpus():
    """Returns list of available GPU ids."""
    output = run_command("nvidia-smi -L")
    # lines of the form GPU 0: TITAN X
    gpu_regex = re.compile(r"GPU (?P<gpu_id>\d+):")
    result = []
    for line in output.strip().split("\n"):
        m = gpu_regex.match(line)
        assert m, "Couldnt parse "+line
        result.append(int(m.group("gpu_id")))
    return result


def gpu_memory_map():
    """Returns map of GPU id to memory allocated on that GPU."""

    output = run_command("nvidia-smi")
    gpu_output = output[output.find("GPU Memory"):]
    # lines of the form
    # |    0      8734    C   python                                       11705MiB |
    memory_regex = re.compile(r"[|]\s+?(?P<gpu_id>\d+)\D+?(?P<pid>\d+).+[ ](?P<gpu_memory>\d+)MiB")
    rows = gpu_output.split("\n")
    result = {gpu_id: 0 for gpu_id in list_available_gpus()}
    for row in gpu_output.split("\n"):
        m = memory_regex.search(row)
        if not m:
            continue
        gpu_id = int(m.group("gpu_id"))
        gpu_memory = int(m.group("gpu_memory"))
        result[gpu_id] += gpu_memory
    return result


def pick_gpu_lowest_memory():
    """Returns GPU with the least allocated memory"""

    memory_gpu_map = [(memory, gpu_id) for (gpu_id, memory) in gpu_memory_map().items()]
    best_memory, best_gpu = sorted(memory_gpu_map)[0]
    return best_gpu


def pick_free_gpus(num_gpus=1):
    """Returns free GPUs with the least allocated memory"""

    memory_gpu_map = [(memory, gpu_id) for (gpu_id, memory) in gpu_memory_map().items()]
    sorted_list = sorted(memory_gpu_map)
    gpu_list = []
    for i in range(num_gpus):
        if sorted_list[i][0] == 0:
            gpu_list.append(sorted_list[i][1])
    
    if len(gpu_list) < num_gpus:
        return -1

    return ','.join(map(str, gpu_list))

def wait_for_free_gpus(num_gpus = 1, wiat_time=300):
    while True:
        res = pick_free_gpus(num_gpus)
        if res == -1:
            print("waiting for free gpus")
            time.sleep(wiat_time)
        else:
            return res