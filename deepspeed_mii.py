import time
import mii

client = mii.client("mistralai/Mistral-7B-v0.1")

start_time = time.perf_counter()

output = client.generate("Deepspeed is", max_new_tokens=256)

end_time = time.perf_counter()
latency = end_time - start_time


print(output)
print('Latency:{latency}')