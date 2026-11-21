# CareWell Support Bot

A small customer-support chatbot for a **fictional medical store**, built as a RAG pipeline with **LangGraph**.
It answers questions about opening hours, delivery, prescriptions, payment and returns **only from the store's own documents**, refuses to give medical advice, and hands the customer to a human when it is unsure.

> The store, phone numbers and policies are invented for this demo. The idea comes from a real family pharmacy, where the same questions come up every day.

## Why this design

A support bot for a medical store has two failure modes that matter: inventing store policies, and giving medical advice. The pipeline is built around avoiding both.

```mermaid
graph TD;
	start([start]) --> safety(safety check)
	safety -.medical advice.-> refuse(refuse + point to pharmacist)
	safety -.store question.-> retrieve(retrieve: cosine similarity)
	retrieve --> rerank(rerank)
	rerank -.relevant.-> generate(generate answer with sources)
	rerank -.not relevant.-> handoff(hand off to staff)
	refuse --> done([end])
	generate --> done
	handoff --> done
```

1. **Safety check**: rule-based filter for dosage, side-effect, interaction and "which medicine" questions. These get a refusal that points to the pharmacist (and mentions emergencies).
2. **Retrieve**: cosine similarity between TF-IDF vectors of the question and each document section (top 6).
3. **Rerank**: reorders the candidates and keeps the best 3. Default is a light lexical reranker. A neural cross-encoder can be switched on (see below).
4. **Relevance gate**: if even the best candidate is too weak, the bot says it doesn't know and gives the store's contact details instead of guessing.
5. **Generate**: Claude writes a short answer using only the retrieved text and cites the file. Without an API key the bot quotes the best passage, so the demo runs with zero setup.

## Run it

```bash
git clone https://github.com/Nitesh-bit1991/carewell-support-bot && cd carewell-support-bot
pip install -r requirements.txt

streamlit run app.py                        # chat UI
python -m support_bot "Is delivery free?"   # command line
```

For written answers instead of quoted passages: `export ANTHROPIC_API_KEY=...`

Optional neural reranker (downloads a model on first use, not covered by the tests or evaluation below):

```bash
pip install -r requirements-neural.txt
python -m support_bot --reranker cross-encoder "Do you sell baby products?"
```

## Evaluation

`python eval/run_eval.py` runs 23 questions: 15 the store can answer, 3 outside its scope, 5 asking for medical advice. It needs no API key and compares cosine-only with cosine + lexical reranker.

| Setup | Top-1 correct source | In top 3 | Correct refusals / hand-offs |
|---|---|---|---|
| Cosine only | 14 / 15 | 15 / 15 | 7 / 8 |
| Cosine + lexical reranker | 14 / 15 | 15 / 15 | 7 / 8 |

**Honest reading of these numbers**

- The test set is tiny and I wrote it myself, so treat it as a sanity check, not a benchmark.
- The lexical reranker did **not** change the results here. I kept it because it is dependency-free, but it is not the reason the bot works.
- **Failure 1:** "Do you sell baby products?" ranks `delivery.md` above `products.md`. TF-IDF favours short sections that repeat the word "products" over a longer section that actually mentions babies. The right document is still in the top 3.
- **Failure 2:** "Do you offer car rental?" is answered with the payment section because the word "offer" matches ("we do not offer purchases on credit"). Scores for this false match overlap with scores of correct answers, so no single threshold fixes it.
- Both failures come from matching on words rather than meaning. The next step would be embedding-based retrieval plus the cross-encoder reranker, evaluated on a larger question set.

## Limitations

- The medical-advice filter is a list of patterns. It will miss unusual phrasings and could over-block some. A real deployment needs a classifier and review by the pharmacist.
- Documents are split by heading, one chunk per section. This works for short help pages but not for long documents.
- This demo must never be used to give actual medical guidance.

## Project layout

```
app.py                  Streamlit chat UI
support_bot/graph.py    LangGraph workflow
support_bot/retrieval.py  cosine retrieval and rerankers
support_bot/safety.py   medical-advice filter
support_bot/generation.py answer writing (Claude or quoted passage)
data/docs/              the store's help documents
eval/                   questions and evaluation script
tests/                  unit tests
```

## Built with AI assistance

I built this project with help from Claude (Anthropic), which drafted much of the code, the store documents and the evaluation questions. I chose the use case and the design direction (LangGraph, cosine retrieval with a reranker, a medical-advice safety check) and ran the project myself.

## License

MIT
