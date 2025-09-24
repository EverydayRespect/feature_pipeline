import time
import pynvml

def monitor_gpu(interval=1.0, duration=600):
    """
    实时监控 GPU 利用率、显存占用。

    Args:
        interval (float): 采样间隔（秒）
        duration (int): 监控总时长（秒）
    """
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()

    print(f"Monitoring {device_count} GPUs every {interval} sec for {duration} sec...\n")

    start = time.time()
    while time.time() - start < duration:
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            print(
                f"[GPU-{i}] Util: {util.gpu:3d}% | Mem: {mem_info.used / 1024**2:.0f} / {mem_info.total / 1024**2:.0f} MB"
            )
        print("-" * 50)
        time.sleep(interval)

    pynvml.nvmlShutdown()


if __name__ == "__main__":
    monitor_gpu(interval=1.0, duration=600)  # 每秒采样一次，总共30秒
