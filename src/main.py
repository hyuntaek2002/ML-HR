from collect import collect_news
from evaluate import run_evaluation


def main():
    print("ML-HR pipeline started")
    print("Step 1/2: collect news")
    collect_news()

    print("Step 2/2: evaluate summaries")
    run_evaluation()

    print("ML-HR pipeline finished")


if __name__ == "__main__":
    main()
