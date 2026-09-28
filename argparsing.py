import argparse
def parse_arg():
    parser = argparse.ArgumentParser(description="A french gpt model")
    #parser.add_argument("--train", action="store_true", help="Train the model") 
    parser.add_argument("-bs", type=int, required=True, help="select batch size")

    return parser.parse_args()

def main():
    args = parse_arg()
    print(f"Batch size: {args.bs}")

if __name__ == "__main__":
    main()
