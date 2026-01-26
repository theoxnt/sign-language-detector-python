from src.cli import ask_user, get_parser
from src.core import inference_classifier
from src.io_ import print_prompt

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    if args.edit:
        print_prompt("\nWelcome to the Sign Language Detector!\n"
                     "What do you want to do?\n")
        ask_user()
    else:
        inference_classifier('f')

if __name__ == "__main__":
    main()
