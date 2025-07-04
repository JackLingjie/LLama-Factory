import subprocess
import time
import tqdm

def get_average_gpu_usage(host):
    """通过 SSH 获取远程主机上 GPU 的平均使用率和内存使用率（不限制 GPU 数量）"""
    try:
        command = f"ssh -o ConnectTimeout=3 {host} LANG=C nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader"
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate()

        stderr_text = error.decode().strip()
        stdout_text = output.decode().strip()

        # 只在非警告时才报错
        if stderr_text and "Warning: Permanently added" not in stderr_text:
            print(f"[ERROR][{host}] SSH error: {stderr_text}")
            return None

        lines = stdout_text.splitlines()

        if len(lines) == 0:
            print(f"[WARN][{host}] No GPU info returned.")
            return None

        total_utilization = 0
        total_memory_used = 0
        total_memory_total = 0
        gpu_count = 0

        for line in lines:
            try:
                values = line.split(", ")
                if len(values) == 3:
                    total_utilization += int(values[0].replace('%', '').strip())
                    total_memory_used += int(values[1].replace('MiB', '').strip())
                    total_memory_total += int(values[2].replace('MiB', '').strip())
                    gpu_count += 1
            except ValueError:
                print(f"[WARN][{host}] Parse error in line: {line}")
                continue

        if gpu_count == 0:
            print(f"[WARN][{host}] No valid GPU data parsed.")
            return None

        average_utilization = total_utilization / gpu_count
        average_memory_used = total_memory_used / gpu_count
        average_memory_total = total_memory_total / gpu_count

        return {
            "average_utilization": f"{average_utilization:.2f}%",
            "average_memory_used": f"{average_memory_used:.2f} MiB",
            "average_memory_total": f"{average_memory_total:.2f} MiB",
        }

    except Exception as e:
        print(f"[EXCEPTION][{host}] {str(e)}")
        return None


def read_hostfile(hostfile_path):
    """从 hostfile 中读取主机列表"""
    try:
        with open(hostfile_path, "r") as f:
            hosts = [line.strip() for line in f.readlines() if line.strip()]
        return hosts
    except FileNotFoundError:
        print(f"[ERROR] Hostfile not found: {hostfile_path}")
        return None

if __name__ == "__main__":
    hostfile_path = "bash_script/auto_run_8/distribute_run_8/node_config/nodes.txt"
    hosts = read_hostfile(hostfile_path)

    if hosts:
        header = "{:<15} | {:<15} | {:<25}".format("Node", "GPU 利用率", "Memory 利用率")
        separator = "-" * 61

        while True:
            lines = []
            for host in tqdm.tqdm(hosts):
                gpu_data = get_average_gpu_usage(host)
                if gpu_data:
                    line = "{:<15} | {:<15} | {:<25}".format(
                        host,
                        gpu_data["average_utilization"],
                        f"{gpu_data['average_memory_used']} / {gpu_data['average_memory_total']}",
                    )
                else:
                    line = "{:<15} | {:<15} | {:<25}".format(host, "无法获取", "无法获取")
                lines.append(line)

            # 清屏并打印表格
            print("\033c", end="")  # 替代原来的 \033[H
            print(header)
            print(separator)
            for line in lines:
                print(line)

            time.sleep(5)
