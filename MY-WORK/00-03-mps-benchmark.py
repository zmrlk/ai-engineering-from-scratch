import sys
import time

import torch

print("Interpreter:", sys.executable)

size = 2000

a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

print("Urządzenie macierzy A:", a_cpu.device)
print("Rozmiar macierzy A:", a_cpu.shape)

a_mps = a_cpu.to("mps")
b_mps = b_cpu.to("mps")

print("A przed przeniesieniem:", a_cpu.device)
print("A po przeniesieniu:", a_mps.device)

start_cpu = time.perf_counter()

c_cpu = a_cpu @ b_cpu

cpu_time = time.perf_counter() - start_cpu

print(f"Czas CPU: {cpu_time:.4f} sekundy")
print("Wynik CPU znajduje się na:", c_cpu.device)
# Rozgrzewka GPU — pierwsze uruchomienie zawiera dodatkowy koszt przygotowania
_ = a_mps @ b_mps
torch.mps.synchronize()

start_mps = time.perf_counter()

c_mps = a_mps @ b_mps

torch.mps.synchronize()
mps_time = time.perf_counter() - start_mps

print(f"Czas MPS: {mps_time:.4f} sekundy")
print("Wynik MPS znajduje się na:", c_mps.device)
print(f"Przyspieszenie GPU: {cpu_time / mps_time:.2f}x")