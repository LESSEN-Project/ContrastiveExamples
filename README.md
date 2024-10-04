## Usage

To run an experiment, you can use the following command with arguments:

```bash
python run_exp.py -d lamp_5_dev -k 5 -f WF DPF -r contriever -ce 3
```

- `--d`: Dataset name. Supported datasets are LaMP (4, 5, 7) and Amazon reviews. LaMP datasets should be in the **lamp_{dataset_num}_{data_split}** format, and they utilize the user-based splits by default. Ex. lamp_4_test, lamp_5_dev. Amazon datasets should be in the **amazon_{category}_{year}** format. Ex: amazon_All_Beauty_2018. Default: lamp_5_dev
- `--k`: Number of retrieved documents for RAG. If None is passed, k is inferred by the length of the user profiles. Default: None
- `--f`: The feature set. Features should be separated by space. Default None
- `--r`: Retriever, can be "contriever", "dpr", or any other model available in SentenceTransformers. Default: contriever
- `--ce`: The number of contrastive users, if None it won't use the contrastive examples method. Default: None

For evaluation:

```bash
python eval.py -d dataset_name
```
