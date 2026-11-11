"""Check the exact two-GPU NCCL communication path without training."""

import argparse
import os
import socket
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing

from scripts.training_metrics import atomic_json


def worker(rank, address, output):
  torch.cuda.set_device(rank)
  distributed.init_process_group("nccl", init_method=address, rank=rank, world_size=2,
                                 timeout=timedelta(seconds=45))
  try:
    value = torch.tensor([rank + 1.], device=f"cuda:{rank}")
    distributed.all_reduce(value)
    if value.item() != 3.:
      raise RuntimeError("GPU collective returned an incorrect sum")
    distributed.barrier()
    atomic_json(Path(output) / f"collective-gpu{rank}.json",
                {"status": "passed", "rank": rank, "sum": value.item(), "backend": "nccl",
                 "nccl_p2p_disable": os.environ["NCCL_P2P_DISABLE"], "optimizer_steps": 0})
  finally:
    distributed.destroy_process_group()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  if torch.cuda.device_count() != 2:
    parser.error("This profile expects exactly two visible GPUs")
  args.output.mkdir(parents=True, exist_ok=True)
  os.environ.setdefault("NCCL_P2P_DISABLE", "1")
  with socket.socket() as listener:
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
  multiprocessing.spawn(worker, args=(f"tcp://127.0.0.1:{port}", str(args.output)), nprocs=2, join=True)
  print("PASS two-GPU NCCL all-reduce and barrier; zero optimizer steps")


if __name__ == "__main__":
  main()
