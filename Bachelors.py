import os
import queue
import time
import random
import matplotlib.pyplot as plt
from typing import Optional

MEMORY_SIZE = 60
MAX_VARIABLES_PER_PROCESS = 3
MAX_LINE_LENGTH = 100
MAX_PROCESSES = 10
TIME_QUANTUM = 1

NUM_CORES = 20 # Number of cores

# Global variables for mutexes
file_mutex = 1
user_input_mutex = 1
screen_output_mutex = 1

DEADLINE_MARGIN = 8 # Define a deadline margin for simplicity

# Modify the PCB class to include a deadline
class PCB:
    def __init__(self, process_id: int):
        self.process_id = process_id
        self.process_state = "READY"
        self.program_counter = 0
        self.memory_lower_bound = 0
        self.memory_upper_bound = MEMORY_SIZE - 1
        self.cycles_remaining = TIME_QUANTUM
        self.waiting_for_resource = ""
        self.start_time = 0
        self.finish_time = 0
        self.waiting_time = 0
        self.turnaround_time = 0
        self.block_start_time = 0  # For tracking block start time
        self.deadline = DEADLINE_MARGIN  # Set a deadline margin for simplicity

class Process:
    def __init__(self, process_id: int, arrival_time: int):
        self.instructions = [""] * MEMORY_SIZE
        self.variables = [""] * MAX_VARIABLES_PER_PROCESS
        self.pcb = PCB(process_id)
        self.arrival_time = arrival_time
        self.pcb.deadline = arrival_time + DEADLINE_MARGIN  # Deadline based on arrival time

class ProcessQueue:
    def __init__(self, max_size: int = MAX_PROCESSES):
        self.queue = queue.Queue(maxsize=max_size)

    def is_empty(self):
        return self.queue.empty()

    def is_full(self):
        return self.queue.full()

    def enqueue(self, process: Process):
        if not self.is_full():
            self.queue.put(process)

    def dequeue(self) -> Optional[Process]:
        if not self.is_empty():
            return self.queue.get()
        return None

    def __iter__(self):
        return iter(list(self.queue.queue))

# Global operating system state
readyQueues = [ProcessQueue() for _ in range(NUM_CORES)]
blockedQueue = ProcessQueue()
storageQueue = ProcessQueue()
clockCycles = 0
next_process_id = 1

# Data collection for plotting graphs and metrics
waiting_times = []
turnaround_times = []
cpu_utilization = []
blocking_times = [[] for _ in range(NUM_CORES)]  # To store blocking times per core
core_utilization = [[] for _ in range(NUM_CORES)]  # To store utilization per core
processes = []

# Constants for Schedulability
  # Simulated deadline margin for processes

def allocate_memory(process: Process, memory_lower_bound: int, memory_upper_bound: int):
    process.pcb.memory_lower_bound = memory_lower_bound
    process.pcb.memory_upper_bound = memory_upper_bound

def store_instructions(process: Process, instruction: str, instruction_index: int):
    process.instructions[instruction_index] = instruction

def store_variables(process: Process, variable: str, value: str):
    for i in range(MAX_VARIABLES_PER_PROCESS):
        if not process.variables[i] or process.variables[i].startswith(f"{variable}="):
            process.variables[i] = f"{variable}={value}"
            break

def retrieve_variable(process: Process, variable_name: str) -> Optional[str]:
    for var_value_pair in process.variables:
        if var_value_pair:
            var, val = var_value_pair.split("=", 1)
            if var == variable_name:
                return val
    return None

def execute_assign(process: Process, variable: str, value: str):
    if value == "input":
        value = input(f"Please enter a value for variable {variable}: ")
    elif value.startswith("readFile"):
        file_var = value.split()[1]
        filename = retrieve_variable(process, file_var)
        if filename and os.path.isfile(filename):
            with open(filename, "r") as file:
                value = file.readline().strip()
        else:
            print(f"Filename variable '{file_var}' not found.")
            return
    store_variables(process, variable, value)

def execute_write_file(process: Process, filename_variable: str, data_variable: str):
    filename = retrieve_variable(process, filename_variable)
    data = retrieve_variable(process, data_variable)
    if filename and data:
        with open(filename, "w") as file:
            file.write(data)
    else:
        print("Error: Invalid filename or data.")

def execute_read_file(process: Process, filename_variable: str):
    filename = retrieve_variable(process, filename_variable)
    if filename and os.path.isfile(filename):
        with open(filename, "r") as file:
            for line in file:
                print(line, end="")
    else:
        print(f"Filename variable '{filename_variable}' not found.")

def execute_print(process: Process, variable: str):
    value = retrieve_variable(process, variable)
    if value:
        print(value)
    else:
        print(f"Variable '{variable}' not found.")

def block_process(process: Process, resource: str):
    global blocking_times
    process.pcb.process_state = "BLOCKED"
    process.pcb.waiting_for_resource = resource
    process.pcb.block_start_time = clockCycles  # Record blocking start time
    blockedQueue.enqueue(process)

def unblock_processes(resource: str):
    global blocking_times
    temp_queue = queue.Queue()
    while not blockedQueue.is_empty():
        process = blockedQueue.dequeue()
        if process.pcb.waiting_for_resource == resource:
            process.pcb.process_state = "READY"
            core_num = process.pcb.process_id % NUM_CORES
            blocking_time = clockCycles - process.pcb.block_start_time  # Calculate blocking time
            blocking_times[core_num].append(blocking_time)  # Log blocking time
            readyQueues[core_num].enqueue(process)
        else:
            temp_queue.put(process)
    while not temp_queue.empty():
        blockedQueue.enqueue(temp_queue.get())

def execute_sem_wait(process: Process, resource: str):
    global user_input_mutex, file_mutex, screen_output_mutex
    if resource == "userInput":
        if user_input_mutex == 0:
            block_process(process, resource)
        else:
            user_input_mutex = 0
    elif resource == "file":
        if file_mutex == 0:
            block_process(process, resource)
        else:
            file_mutex = 0
    elif resource == "userOutput":
        if screen_output_mutex == 0:
            block_process(process, resource)
        else:
            screen_output_mutex = 0

def execute_sem_signal(resource: str):
    global user_input_mutex, file_mutex, screen_output_mutex
    if resource == "userInput":
        user_input_mutex = 1
    elif resource == "file":
        file_mutex = 1
    elif resource == "userOutput":
        screen_output_mutex = 1
    unblock_processes(resource)

def execute_print_from_to(process: Process, start_var: str, end_var: str):
    start_str = retrieve_variable(process, start_var)
    end_str = retrieve_variable(process, end_var)
    if start_str and end_str:
        start = int(start_str)
        end = int(end_str)
        print(" ".join(str(i) for i in range(start, end + 1)))
    else:
        print("Error: Variables not found.")

def execute_process(process: Process):
    global clockCycles
    process.pcb.process_state = "RUNNING"
    line = process.instructions[process.pcb.program_counter]
    print(f"Executing instruction [{line}] from Process {process.pcb.process_id} at clock cycle {clockCycles}")

    instruction_parts = line.split()
    instruction = instruction_parts[0]

    if instruction == "print":
        execute_print(process, instruction_parts[1])
    elif instruction == "assign":
        if len(instruction_parts) > 3:
            combined_value = " ".join(instruction_parts[2:])
            execute_assign(process, instruction_parts[1], combined_value)
        else:
            execute_assign(process, instruction_parts[1], instruction_parts[2])
    elif instruction == "writeFile":
        execute_write_file(process, instruction_parts[1], instruction_parts[2])
    elif instruction == "readFile":
        execute_read_file(process, instruction_parts[1])
    elif instruction == "printFromTo":
        execute_print_from_to(process, instruction_parts[1], instruction_parts[2])
    elif instruction == "semWait":
        execute_sem_wait(process, instruction_parts[1])
    elif instruction == "semSignal":
        execute_sem_signal(instruction_parts[1])
    else:
        print(f"Unknown instruction: {instruction}")

    process.instructions[process.pcb.program_counter] = ""
    process.pcb.program_counter += 1
    process.pcb.cycles_remaining -= 1

    if process.pcb.process_state == "RUNNING":
        if process.pcb.program_counter >= MEMORY_SIZE or not process.instructions[process.pcb.program_counter]:
            process.pcb.process_state = "FINISHED"
            process.pcb.finish_time = clockCycles
            process.pcb.turnaround_time = process.pcb.finish_time - process.arrival_time
            process.pcb.waiting_time = process.pcb.turnaround_time - (TIME_QUANTUM * (process.pcb.program_counter // TIME_QUANTUM))
            turnaround_times.append(process.pcb.turnaround_time)
            waiting_times.append(process.pcb.waiting_time)
        elif process.pcb.cycles_remaining == 0:
            process.pcb.process_state = "READY"
            process.pcb.cycles_remaining = TIME_QUANTUM
            core_num = process.pcb.process_id % NUM_CORES
            readyQueues[core_num].enqueue(process)
        else:
            process.pcb.process_state = "READY"

def execute_program(filename: str, process: Process) -> bool:
    if not os.path.isfile(filename):
        print(f"Error opening file: {filename}")
        return False

    with open(filename, "r") as file:
        instruction_index = 0
        for line in file:
            store_instructions(process, line.strip(), instruction_index)
            instruction_index += 1

    return True

def calculate_average_delay_per_task(process):
    return process.pcb.turnaround_time

def calculate_average_blocked_delay_per_task(process):
    blocked_time = sum(blocking_time for blocking_time in blocking_times[process.pcb.process_id % NUM_CORES])
    return blocked_time



def print_queue(queueName: str, queue: list):
    print(f"{queueName} Queue:")
    print("+------------+-----------------------+")
    print("| Process ID |   Program Counter     |")
    print("+------------+-----------------------+")
    for process in queue:
        if process.pcb.process_state != "FINISHED":
            print(f"| {process.pcb.process_id:<10} | {process.pcb.program_counter:<21} |")
    print("+------------+-----------------------+")

def print_storage_unit(queue: list):
    print("Storage Unit:")
    print("+------------+-----------------------+")
    print("| Process ID |   Instructions          |")
    print("+------------+-----------------------+")
    for process in queue:
        for instruction in process.instructions:
            if instruction:
                print(f"| {process.pcb.process_id:<10} | {instruction:<21} |")
    print("+------------+-----------------------+")

def assign_process_to_core(process: Process, core_num: int):
    readyQueues[core_num].enqueue(process)
    print(f"Process {process.pcb.process_id} assigned to Core {core_num} has arrived at clock cycle {clockCycles}")
    print_queue(f"Ready (Core {core_num})", list(readyQueues[core_num].queue.queue))

def simulate_process_management(arrival_times, filenames):
    global clockCycles, next_process_id, readyQueues, blockedQueue, storageQueue, cpu_utilization, core_utilization, processes

    # Reset the queues and clock cycles
    readyQueues = [ProcessQueue() for _ in range(NUM_CORES)]
    blockedQueue = ProcessQueue()
    storageQueue = ProcessQueue()
    clockCycles = 0
    next_process_id = 1
    core_utilization = [[] for _ in range(NUM_CORES)]  # Reset core utilization
    processes = []  # Reset processes list

    process_count = 0
    acceptance_ratios = []

    for i in range(len(arrival_times)):
        arrival_time = arrival_times[i]
        filename = filenames[i]

        process = Process(next_process_id, arrival_time)
        next_process_id += 1

        if not execute_program(filename, process):
            return

        processes.append(process)
        process_count += 1

    executedProcess = None

    while True:
        anyProcessActive = False

        for i in range(process_count):
            if arrival_times[i] == clockCycles:
                storageQueue.enqueue(processes[i])
                assign_process_to_core(processes[i], i % NUM_CORES)

        for core_num in range(NUM_CORES):
            core_was_active = False
            while not readyQueues[core_num].is_empty():
                process = readyQueues[core_num].dequeue()
                if process.pcb.process_state == "READY":
                    anyProcessActive = True
                    core_was_active = True
                    executedProcess = process
                    execute_process(process)
                    core_utilization[core_num].append(1)  # Core was active
                    clockCycles += 1

                    for j in range(process_count):
                        if arrival_times[j] == clockCycles:
                            storageQueue.enqueue(processes[j])
                            assign_process_to_core(processes[j], j % NUM_CORES)

                    if executedProcess:
                        temp_queue = queue.Queue()
                        while not storageQueue.is_empty():
                            storageProcess = storageQueue.dequeue()
                            if storageProcess.pcb.process_id == executedProcess.pcb.process_id:
                                storageProcess.instructions[executedProcess.pcb.program_counter - 1] = ""
                            temp_queue.put(storageProcess)

                        while not temp_queue.empty():
                            storageQueue.enqueue(temp_queue.get())
                        executedProcess = None

                    print_queue(f"Ready (Core {core_num})", list(readyQueues[core_num].queue.queue))
                    print_queue("Blocked", list(blockedQueue.queue.queue))
                    print_storage_unit(list(storageQueue.queue.queue))

                    if process.pcb.process_state == "READY":
                        readyQueues[core_num].enqueue(process)
                    elif process.pcb.process_state in ["BLOCKED", "FINISHED"]:
                        if process.pcb.process_state == "FINISHED":
                            print(f"Process {process.pcb.process_id} has finished execution.")
                            temp_queue = queue.Queue()
                            while not storageQueue.is_empty():
                                storageProcess = storageQueue.dequeue()
                                if storageProcess.pcb.process_id != process.pcb.process_id:
                                    temp_queue.put(storageProcess)
                            while not temp_queue.empty():
                                storageQueue.enqueue(temp_queue.get())

            if not core_was_active:
                core_utilization[core_num].append(0)  # Core was idle

        print_queue("Blocked", list(blockedQueue.queue.queue))
        print_storage_unit(list(storageQueue.queue.queue))

        if not anyProcessActive and blockedQueue.is_empty():
            break

        if not anyProcessActive:
            cpu_utilization.append((clockCycles, 0))
            clockCycles += 1
        else:
            cpu_utilization.append((clockCycles, 1))

        acceptance_ratio = calculate_acceptance_ratio()
        acceptance_ratios.append(acceptance_ratio)
        time.sleep(1)  # Pause for one second

    print("All processes have finished execution.")
    return acceptance_ratios
def simulate_process_management_pip(arrival_times, filenames):
    global clockCycles, next_process_id, readyQueues, blockedQueue, storageQueue, cpu_utilization, core_utilization, processes

    # Reset the queues and clock cycles
    readyQueues = [ProcessQueue() for _ in range(NUM_CORES)]
    blockedQueue = ProcessQueue()
    storageQueue = ProcessQueue()
    clockCycles = 0
    next_process_id = 1
    core_utilization = [[] for _ in range(NUM_CORES)]  # Reset core utilization
    processes = []  # Reset processes list

    process_count = 0
    acceptance_ratios = []

    for i in range(len(arrival_times)):
        arrival_time = arrival_times[i]
        filename = filenames[i]

        process = Process(next_process_id, arrival_time)
        next_process_id += 1

        if not execute_program(filename, process):
            return

        processes.append(process)
        process_count += 1

    executedProcess = None

    while True:
        anyProcessActive = False

        for i in range(process_count):
            if arrival_times[i] == clockCycles:
                storageQueue.enqueue(processes[i])
                assign_process_to_core(processes[i], i % NUM_CORES)

        for core_num in range(NUM_CORES):
            core_was_active = False
            while not readyQueues[core_num].is_empty():
                process = readyQueues[core_num].dequeue()
                if process.pcb.process_state == "READY":
                    anyProcessActive = True
                    core_was_active = True
                    executedProcess = process
                    execute_process(process)
                    core_utilization[core_num].append(1)  # Core was active
                    clockCycles += 1

                    for j in range(process_count):
                        if arrival_times[j] == clockCycles:
                            storageQueue.enqueue(processes[j])
                            assign_process_to_core(processes[j], j % NUM_CORES)

                    if executedProcess:
                        temp_queue = queue.Queue()
                        while not storageQueue.is_empty():
                            storageProcess = storageQueue.dequeue()
                            if storageProcess.pcb.process_id == executedProcess.pcb.process_id:
                                storageProcess.instructions[executedProcess.pcb.program_counter - 1] = ""
                            temp_queue.put(storageProcess)

                        while not temp_queue.empty():
                            storageQueue.enqueue(temp_queue.get())
                        executedProcess = None

                    print_queue(f"Ready (Core {core_num})", list(readyQueues[core_num].queue.queue))
                    print_queue("Blocked", list(blockedQueue.queue.queue))
                    print_storage_unit(list(storageQueue.queue.queue))

                    if process.pcb.process_state == "READY":
                        readyQueues[core_num].enqueue(process)
                    elif process.pcb.process_state in ["BLOCKED", "FINISHED"]:
                        if process.pcb.process_state == "FINISHED":
                            print(f"Process {process.pcb.process_id} has finished execution.")
                            temp_queue = queue.Queue()
                            while not storageQueue.is_empty():
                                storageProcess = storageQueue.dequeue()
                                if storageProcess.pcb.process_id != process.pcb.process_id:
                                    temp_queue.put(storageProcess)
                            while not temp_queue.empty():
                                storageQueue.enqueue(temp_queue.get())

            if not core_was_active:
                core_utilization[core_num].append(0)  # Core was idle

        print_queue("Blocked", list(blockedQueue.queue.queue))
        print_storage_unit(list(storageQueue.queue.queue))

        if not anyProcessActive and blockedQueue.is_empty():
            break

        if not anyProcessActive:
            cpu_utilization.append((clockCycles, 0))
            clockCycles += 1
        else:
            cpu_utilization.append((clockCycles, 1))

        acceptance_ratio = calculate_acceptance_ratio()
        acceptance_ratios.append(acceptance_ratio)
        time.sleep(1)  # Pause for one second

    print("All processes have finished execution.")
    return acceptance_ratios


def simulate_process_management_mpcp(arrival_times, filenames):
    global clockCycles, next_process_id, readyQueues, blockedQueue, storageQueue, cpu_utilization, core_utilization, processes

    # Reset the queues and clock cycles
    readyQueues = [ProcessQueue() for _ in range(NUM_CORES)]
    blockedQueue = ProcessQueue()
    storageQueue = ProcessQueue()
    clockCycles = 0
    next_process_id = 1
    core_utilization = [[] for _ in range(NUM_CORES)]  # Reset core utilization
    processes = []  # Reset processes list

    process_count = 0
    acceptance_ratios = []

    for i in range(len(arrival_times)):
        arrival_time = arrival_times[i]
        filename = filenames[i]

        process = Process(next_process_id, arrival_time)
        next_process_id += 1

        if not execute_program(filename, process):
            return

        processes.append(process)
        process_count += 1

    executedProcess = None

    while True:
        anyProcessActive = False

        for i in range(process_count):
            if arrival_times[i] == clockCycles:
                storageQueue.enqueue(processes[i])
                assign_process_to_core(processes[i], i % NUM_CORES)

        for core_num in range(NUM_CORES):
            core_was_active = False
            while not readyQueues[core_num].is_empty():
                process = readyQueues[core_num].dequeue()
                if process.pcb.process_state == "READY":
                    anyProcessActive = True
                    core_was_active = True
                    executedProcess = process
                    execute_process(process)
                    core_utilization[core_num].append(1)  # Core was active
                    clockCycles += 1

                    for j in range(process_count):
                        if arrival_times[j] == clockCycles:
                            storageQueue.enqueue(processes[j])
                            assign_process_to_core(processes[j], j % NUM_CORES)

                    if executedProcess:
                        temp_queue = queue.Queue()
                        while not storageQueue.is_empty():
                            storageProcess = storageQueue.dequeue()
                            if storageProcess.pcb.process_id == executedProcess.pcb.process_id:
                                storageProcess.instructions[executedProcess.pcb.program_counter - 1] = ""
                            temp_queue.put(storageProcess)

                        while not temp_queue.empty():
                            storageQueue.enqueue(temp_queue.get())
                        executedProcess = None

                    print_queue(f"Ready (Core {core_num})", list(readyQueues[core_num].queue.queue))
                    print_queue("Blocked", list(blockedQueue.queue.queue))
                    print_storage_unit(list(storageQueue.queue.queue))

                    if process.pcb.process_state == "READY":
                        readyQueues[core_num].enqueue(process)
                    elif process.pcb.process_state in ["BLOCKED", "FINISHED"]:
                        if process.pcb.process_state == "FINISHED":
                            print(f"Process {process.pcb.process_id} has finished execution.")
                            temp_queue = queue.Queue()
                            while not storageQueue.is_empty():
                                storageProcess = storageQueue.dequeue()
                                if storageProcess.pcb.process_id != process.pcb.process_id:
                                    temp_queue.put(storageProcess)
                            while not temp_queue.empty():
                                storageQueue.enqueue(temp_queue.get())

            if not core_was_active:
                core_utilization[core_num].append(0)  # Core was idle

        print_queue("Blocked", list(blockedQueue.queue.queue))
        print_storage_unit(list(storageQueue.queue.queue))

        if not anyProcessActive and blockedQueue.is_empty():
            break

        if not anyProcessActive:
            cpu_utilization.append((clockCycles, 0))
            clockCycles += 1
        else:
            cpu_utilization.append((clockCycles, 1))

        acceptance_ratio = calculate_acceptance_ratio()
        acceptance_ratios.append(acceptance_ratio)
        time.sleep(1)  # Pause for one second

    print("All processes have finished execution.")
    return acceptance_ratios

def simulate_process_management_r_pcp(arrival_times, filenames):
    global clockCycles, next_process_id, readyQueues, blockedQueue, storageQueue, cpu_utilization, core_utilization, processes

    # Reset the queues and clock cycles
    readyQueues = [ProcessQueue() for _ in range(NUM_CORES)]
    blockedQueue = ProcessQueue()
    storageQueue = ProcessQueue()
    clockCycles = 0
    next_process_id = 1
    core_utilization = [[] for _ in range(NUM_CORES)]  # Reset core utilization
    processes = []  # Reset processes list

    process_count = 0
    acceptance_ratios = []

    for i in range(len(arrival_times)):
        arrival_time = arrival_times[i]
        filename = filenames[i]

        process = Process(next_process_id, arrival_time)
        next_process_id += 1

        if not execute_program(filename, process):
            return

        processes.append(process)
        process_count += 1

    executedProcess = None

    while True:
        anyProcessActive = False

        for i in range(process_count):
            if arrival_times[i] == clockCycles:
                storageQueue.enqueue(processes[i])
                assign_process_to_core(processes[i], i % NUM_CORES)

        for core_num in range(NUM_CORES):
            core_was_active = False
            while not readyQueues[core_num].is_empty():
                process = readyQueues[core_num].dequeue()
                if process.pcb.process_state == "READY":
                    anyProcessActive = True
                    core_was_active = True
                    executedProcess = process
                    execute_process(process)
                    core_utilization[core_num].append(1)  # Core was active
                    clockCycles += 1

                    for j in range(process_count):
                        if arrival_times[j] == clockCycles:
                            storageQueue.enqueue(processes[j])
                            assign_process_to_core(processes[j], j % NUM_CORES)

                    if executedProcess:
                        temp_queue = queue.Queue()
                        while not storageQueue.is_empty():
                            storageProcess = storageQueue.dequeue()
                            if storageProcess.pcb.process_id == executedProcess.pcb.process_id:
                                storageProcess.instructions[executedProcess.pcb.program_counter - 1] = ""
                            temp_queue.put(storageProcess)

                        while not temp_queue.empty():
                            storageQueue.enqueue(temp_queue.get())
                        executedProcess = None

                    print_queue(f"Ready (Core {core_num})", list(readyQueues[core_num].queue.queue))
                    print_queue("Blocked", list(blockedQueue.queue.queue))
                    print_storage_unit(list(storageQueue.queue.queue))

                    if process.pcb.process_state == "READY":
                        readyQueues[core_num].enqueue(process)
                    elif process.pcb.process_state in ["BLOCKED", "FINISHED"]:
                        if process.pcb.process_state == "FINISHED":
                            print(f"Process {process.pcb.process_id} has finished execution.")
                            temp_queue = queue.Queue()
                            while not storageQueue.is_empty():
                                storageProcess = storageQueue.dequeue()
                                if storageProcess.pcb.process_id != process.pcb.process_id:
                                    temp_queue.put(storageProcess)
                            while not temp_queue.empty():
                                storageQueue.enqueue(temp_queue.get())

            if not core_was_active:
                core_utilization[core_num].append(0)  # Core was idle

        print_queue("Blocked", list(blockedQueue.queue.queue))
        print_storage_unit(list(storageQueue.queue.queue))

        if not anyProcessActive and blockedQueue.is_empty():
            break

        if not anyProcessActive:
            cpu_utilization.append((clockCycles, 0))
            clockCycles += 1
        else:
            cpu_utilization.append((clockCycles, 1))

        acceptance_ratio = calculate_acceptance_ratio()
        acceptance_ratios.append(acceptance_ratio)
        time.sleep(1)  # Pause for one second

    print("All processes have finished execution.")
    return acceptance_ratios

def simulate_process_management_r_npp(arrival_times, filenames):
    global clockCycles, next_process_id, readyQueues, blockedQueue, storageQueue, cpu_utilization, core_utilization, processes

    # Reset the queues and clock cycles
    readyQueues = [ProcessQueue() for _ in range(NUM_CORES)]
    blockedQueue = ProcessQueue()
    storageQueue = ProcessQueue()
    clockCycles = 0
    next_process_id = 1
    core_utilization = [[] for _ in range(NUM_CORES)]  # Reset core utilization
    processes = []  # Reset processes list

    process_count = 0
    acceptance_ratios = []

    for i in range(len(arrival_times)):
        arrival_time = arrival_times[i]
        filename = filenames[i]

        process = Process(next_process_id, arrival_time)
        next_process_id += 1

        if not execute_program(filename, process):
            return

        processes.append(process)
        process_count += 1

    executedProcess = None

    while True:
        anyProcessActive = False

        for i in range(process_count):
            if arrival_times[i] == clockCycles:
                storageQueue.enqueue(processes[i])
                assign_process_to_core(processes[i], i % NUM_CORES)

        for core_num in range(NUM_CORES):
            core_was_active = False
            while not readyQueues[core_num].is_empty():
                process = readyQueues[core_num].dequeue()
                if process.pcb.process_state == "READY":
                    anyProcessActive = True
                    core_was_active = True
                    executedProcess = process
                    execute_process(process)
                    core_utilization[core_num].append(1)  # Core was active
                    clockCycles += 1

                    for j in range(process_count):
                        if arrival_times[j] == clockCycles:
                            storageQueue.enqueue(processes[j])
                            assign_process_to_core(processes[j], j % NUM_CORES)

                    if executedProcess:
                        temp_queue = queue.Queue()
                        while not storageQueue.is_empty():
                            storageProcess = storageQueue.dequeue()
                            if storageProcess.pcb.process_id == executedProcess.pcb.process_id:
                                storageProcess.instructions[executedProcess.pcb.program_counter - 1] = ""
                            temp_queue.put(storageProcess)

                        while not temp_queue.empty():
                            storageQueue.enqueue(temp_queue.get())
                        executedProcess = None

                    print_queue(f"Ready (Core {core_num})", list(readyQueues[core_num].queue.queue))
                    print_queue("Blocked", list(blockedQueue.queue.queue))
                    print_storage_unit(list(storageQueue.queue.queue))

                    if process.pcb.process_state == "READY":
                        readyQueues[core_num].enqueue(process)
                    elif process.pcb.process_state in ["BLOCKED", "FINISHED"]:
                        if process.pcb.process_state == "FINISHED":
                            print(f"Process {process.pcb.process_id} has finished execution.")
                            temp_queue = queue.Queue()
                            while not storageQueue.is_empty():
                                storageProcess = storageQueue.dequeue()
                                if storageProcess.pcb.process_id != process.pcb.process_id:
                                    temp_queue.put(storageProcess)
                            while not temp_queue.empty():
                                storageQueue.enqueue(temp_queue.get())

            if not core_was_active:
                core_utilization[core_num].append(0)  # Core was idle

        print_queue("Blocked", list(blockedQueue.queue.queue))
        print_storage_unit(list(storageQueue.queue.queue))

        if not anyProcessActive and blockedQueue.is_empty():
            break

        if not anyProcessActive:
            cpu_utilization.append((clockCycles, 0))
            clockCycles += 1
        else:
            cpu_utilization.append((clockCycles, 1))

        acceptance_ratio = calculate_acceptance_ratio()
        acceptance_ratios.append(acceptance_ratio)
        time.sleep(1)  # Pause for one second

    print("All processes have finished execution.")
    return acceptance_ratios




def simulate_process_management_mrsp(arrival_times, filenames):
    global clockCycles, next_process_id, readyQueues, blockedQueue, storageQueue, cpu_utilization, core_utilization, processes

    # Reset the queues and clock cycles
    readyQueues = [ProcessQueue() for _ in range(NUM_CORES)]
    blockedQueue = ProcessQueue()
    storageQueue = ProcessQueue()
    clockCycles = 0
    next_process_id = 1
    core_utilization = [[] for _ in range(NUM_CORES)]  # Reset core utilization
    processes = []  # Reset processes list

    process_count = 0
    acceptance_ratios = []

    for i in range(len(arrival_times)):
        arrival_time = arrival_times[i]
        filename = filenames[i]

        process = Process(next_process_id, arrival_time)
        next_process_id += 1

        if not execute_program(filename, process):
            return

        processes.append(process)
        process_count += 1

    executedProcess = None

    while True:
        anyProcessActive = False

        for i in range(process_count):
            if arrival_times[i] == clockCycles:
                storageQueue.enqueue(processes[i])
                assign_process_to_core(processes[i], i % NUM_CORES)

        for core_num in range(NUM_CORES):
            core_was_active = False
            while not readyQueues[core_num].is_empty():
                process = readyQueues[core_num].dequeue()
                if process.pcb.process_state == "READY":
                    anyProcessActive = True
                    core_was_active = True
                    executedProcess = process
                    execute_process(process)
                    core_utilization[core_num].append(1)  # Core was active
                    clockCycles += 1

                    for j in range(process_count):
                        if arrival_times[j] == clockCycles:
                            storageQueue.enqueue(processes[j])
                            assign_process_to_core(processes[j], j % NUM_CORES)

                    if executedProcess:
                        temp_queue = queue.Queue()
                        while not storageQueue.is_empty():
                            storageProcess = storageQueue.dequeue()
                            if storageProcess.pcb.process_id == executedProcess.pcb.process_id:
                                storageProcess.instructions[executedProcess.pcb.program_counter - 1] = ""
                            temp_queue.put(storageProcess)

                        while not temp_queue.empty():
                            storageQueue.enqueue(temp_queue.get())
                        executedProcess = None

                    print_queue(f"Ready (Core {core_num})", list(readyQueues[core_num].queue.queue))
                    print_queue("Blocked", list(blockedQueue.queue.queue))
                    print_storage_unit(list(storageQueue.queue.queue))

                    if process.pcb.process_state == "READY":
                        readyQueues[core_num].enqueue(process)
                    elif process.pcb.process_state in ["BLOCKED", "FINISHED"]:
                        if process.pcb.process_state == "FINISHED":
                            print(f"Process {process.pcb.process_id} has finished execution.")
                            temp_queue = queue.Queue()
                            while not storageQueue.is_empty():
                                storageProcess = storageQueue.dequeue()
                                if storageProcess.pcb.process_id != process.pcb.process_id:
                                    temp_queue.put(storageProcess)
                            while not temp_queue.empty():
                                storageQueue.enqueue(temp_queue.get())

            if not core_was_active:
                core_utilization[core_num].append(0)  # Core was idle

        print_queue("Blocked", list(blockedQueue.queue.queue))
        print_storage_unit(list(storageQueue.queue.queue))

        if not anyProcessActive and blockedQueue.is_empty():
            break

        if not anyProcessActive:
            cpu_utilization.append((clockCycles, 0))
            clockCycles += 1
        else:
            cpu_utilization.append((clockCycles, 1))

        acceptance_ratio = calculate_acceptance_ratio()
        acceptance_ratios.append(acceptance_ratio)
        time.sleep(1)  # Pause for one second

    print("All processes have finished execution.")
    return acceptance_ratios






def calculate_speedup_factor(algorithm: str, turnaround_time: int) -> float:
    if algorithm == "RSP":
        return turnaround_times[0] / turnaround_time
    elif algorithm == "MNP":
        return turnaround_times[1] / turnaround_time
    else:
        return 0.0  # Unknown algorithm

def calculate_acceptance_ratio(algorithm: str) -> float:
    if algorithm == "RSP":
        return len(readyQueues[0].queue.queue) / len(turnaround_times)
    elif algorithm == "MNP":
        return len(blockedQueue.queue.queue) / len(turnaround_times)
    else:
        return 0.0  # Unknown algorithm

def check_schedulability():
    schedulable_processes = 0
    for process in processes:
        if process.pcb.finish_time <= process.arrival_time + DEADLINE_MARGIN:
            schedulable_processes += 1
    schedulability = (schedulable_processes / len(processes)) * 100
    return schedulability

def calculate_acceptance_ratio():
    total_processes = len(processes)
    processes_meeting_deadlines = sum(1 for process in processes if process.pcb.finish_time <= process.pcb.deadline)
    acceptance_ratio = (processes_meeting_deadlines / total_processes) * 100
    return acceptance_ratio

def calculate_blocking_ratio():
    blocking_ratios = []
    for core_num in range(NUM_CORES):
        total_tasks = len(core_utilization[core_num])
        blocked_tasks = len(blocking_times[core_num])
        if total_tasks > 0:
            ratio = blocked_tasks / total_tasks
        else:
            ratio = 0
        blocking_ratios.append(ratio)
    return blocking_ratios

def plot_blocking_ratio(blocking_ratios):
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(blocking_ratios)), blocking_ratios, color='skyblue')
    plt.xlabel('Core Number')
    plt.ylabel('Blocking Ratio')
    plt.title('Blocking Ratio per Core')
    plt.show()

def plot_average_delay_per_task(processes):
    average_delays = [calculate_average_delay_per_task(process) for process in processes]
    plt.figure(figsize=(12, 8))
    plt.bar(range(1, len(processes) + 1), average_delays, color='skyblue')
    plt.xlabel('Task ID')
    plt.ylabel('Clock Cycles')
    plt.title('Average Delay for Each Task')
    plt.xticks(range(1, len(processes) + 1))
    plt.show()

def plot_average_blocked_delay_per_task(processes):
    average_blocked_delays = [calculate_average_blocked_delay_per_task(process) for process in processes]
    plt.figure(figsize=(12, 8))
    plt.bar(range(1, len(processes) + 1), average_blocked_delays, color='salmon')
    plt.xlabel('Task ID')
    plt.ylabel('Blocked Clock Cycles')
    plt.title('Average Blocked Delay for Each Task')
    plt.xticks(range(1, len(processes) + 1))
    plt.show()



def plot_graphs(acceptance_ratios):
    # Plot waiting times
    plt.figure(figsize=(10, 8))
    plt.plot(range(len(waiting_times)), waiting_times, marker='o')
    plt.title('Process Waiting Times')
    plt.xlabel('Process')
    plt.ylabel('Waiting Time (clock cycles)')
    plt.tight_layout()
    plt.savefig('waiting_times.png')
    plt.show()

    # Plot turnaround times
    plt.figure(figsize=(10, 8))
    plt.plot(range(len(turnaround_times)), turnaround_times, marker='o')
    plt.title('Process Turnaround Times')
    plt.xlabel('Process')
    plt.ylabel('Turnaround Time (clock cycles)')
    plt.tight_layout()
    plt.savefig('turnaround_times.png')
    plt.show()

    # Plot CPU utilization
    cpu_cycles, utilization = zip(*cpu_utilization)
    plt.figure(figsize=(10, 8))
    plt.plot(cpu_cycles, utilization, drawstyle='steps-pre')
    plt.title('CPU Utilization Over Time')
    plt.xlabel('Clock Cycle')
    plt.ylabel('CPU Utilization (0 or 1)')
    plt.tight_layout()
    plt.savefig('cpu_utilization.png')
    plt.show()

    # Plot blocking times
    plt.figure(figsize=(10, 8))
    for core_num in range(NUM_CORES):
        plt.hist(blocking_times[core_num], alpha=0.5, label=f'Core {core_num}')
    plt.title('Blocking Times Distribution')
    plt.xlabel('Blocking Time (clock cycles)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig('blocking_times.png')
    plt.show()

    # Plot core utilization for each core
    for core_num in range(NUM_CORES):
        core_cycles, core_util = zip(*[(i, u) for i, u in enumerate(core_utilization[core_num])])
        plt.figure(figsize=(10, 8))
        plt.plot(core_cycles, core_util, drawstyle='steps-pre')
        plt.title(f'Core {core_num} Utilization Over Time')
        plt.xlabel('Clock Cycle')
        plt.ylabel('Utilization (0 or 1)')
        plt.tight_layout()
        plt.savefig(f'core_{core_num}_utilization.png')
        plt.show()

    # Plot acceptance ratio
    plt.figure(figsize=(10, 8))
    plt.plot(range(len(acceptance_ratios)), acceptance_ratios, marker='o')
    plt.title('Acceptance Ratio Over Time')
    plt.xlabel('Process')
    plt.ylabel('Acceptance Ratio (%)')
    plt.tight_layout()
    plt.savefig('acceptance_ratio.png')
    plt.show()

def main():
    
    import sys
    processes = []
    global clockCycles, next_process_id

    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <arrival_time1> <program_file1> [<arrival_time2> <program_file2> ...]")
        return

    arrival_times = []
    filenames = []

    for i in range(1, len(sys.argv), 2):
        if i + 1 >= len(sys.argv):
            print(f"Error: Missing program file for arrival time {sys.argv[i]}")
            return

        arrival_time = int(sys.argv[i])
        filename = sys.argv[i + 1]

        arrival_times.append(arrival_time)
        filenames.append(filename)
        for i in range(len(arrival_times)):
         process = Process(next_process_id, arrival_times[i])
        next_process_id += 1

        if not execute_program(filenames[i], process):
            return

        processes.append(process)
        
        for process in processes:
         avg_delay = calculate_average_delay_per_task(process)
        avg_blocked_delay = calculate_average_blocked_delay_per_task(process)
        print(f"Task {process.pcb.process_id}: Average Delay - {avg_delay}, Average Blocked Delay - {avg_blocked_delay}")

    # Plot the average delay and average blocked delay per task
   
  

    acceptance_ratios = simulate_process_management(arrival_times, filenames)
    schedulability = check_schedulability()  # Calculate schedulability
    print(f"Schedulability: {schedulability:.2f}%")
    acceptance_ratio = calculate_acceptance_ratio()  # Calculate acceptance ratio
    print(f"Acceptance Ratio: {acceptance_ratio:.2f}%")
    plot_average_blocked_delay_per_task(processes)
    plot_graphs(acceptance_ratios)
    blocking_ratios = calculate_blocking_ratio()
    plot_blocking_ratio(blocking_ratios)
      # List to hold Process objects

    

    # Calculate and plot average delay and average blocked delay per task
   

if __name__ == "__main__":
    main()
