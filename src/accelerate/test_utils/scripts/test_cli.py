import torch


def main():
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"Successfully ran on {num_gpus} GPUs")


if __name__ == "__main__":
    main()
