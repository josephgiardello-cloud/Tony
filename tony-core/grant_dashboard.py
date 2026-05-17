import argparse

from tony.dashboard import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the TONY dashboard")
    parser.add_argument("--input", required=True, help="Path to a scored JSON file")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    options = parser.parse_args()
    main(options.input, options.host, options.port)
