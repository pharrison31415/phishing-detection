from src.baseline_models import main as baseline_main
from src.baseline_no_engineering import main as baseline_no_engineering_main
from src.bert_model import main as bert_main


if __name__ == "__main__":
    print('\nBaseline models (no engineering):')
    baseline_no_engineering_main()
    print('Baseline models:')
    baseline_main()
    print('\nBERT models:')
    bert_main()