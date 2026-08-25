import torch, time, sys
gb = float(sys.argv[1]); x = torch.empty(int(gb * 2**30), dtype=torch.uint8, device="cuda"); x.fill_(1)
print("holding", gb, "GB", flush=True); time.sleep(float(sys.argv[2]))
