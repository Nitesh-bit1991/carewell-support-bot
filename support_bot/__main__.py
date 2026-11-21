import argparse

from .graph import SupportBot


def main() -> None:
    ap = argparse.ArgumentParser(description="CareWell Medical Store support bot")
    ap.add_argument("question", nargs="*", help="Ask one question (omit for interactive mode)")
    ap.add_argument("--reranker", default="lexical", choices=["none", "lexical", "cross-encoder"])
    args = ap.parse_args()

    bot = SupportBot(reranker=args.reranker)
    if args.question:
        print(bot.ask(" ".join(args.question)).text)
        return
    print("CareWell support bot. Type a question (Ctrl+C to quit).")
    while True:
        try:
            print(bot.ask(input("> ")).text, end="\n\n")
        except (KeyboardInterrupt, EOFError):
            break


if __name__ == "__main__":
    main()
