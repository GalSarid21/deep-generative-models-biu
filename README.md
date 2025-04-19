# Deep Generative Models – Final Project (BIU, 2025)
This repository contains the code for the final project of the `Deep Generative Models` course, taught by `Dr. Ethan Fetaya` at `Bar-Ilan University` (Isreal, 2025).

The project explores how `State Space Models` (SSMs) handle large-context inputs and compares their performance to `Large Language Models` (LLMs), building upon the findings from the `Lost in the Middle` paper (see full citation at the `References` section).

## Table of Contents:

- [Deep Generative Models – Final Project (BIU, 2025)](#deep-generative-models--final-project-biu-2025)
  - [Table of contents](#table-of-contents)
  - [State Space Models](#state-space-models)
  - [Experiment Data](#experiment-data)
  - [Experiment Description](#experiment-description)
  - [Evaluation](#evaluation)
  - [Results](#results)
  - [Future Work](#future-work)
  - [References](#references)

## State Space Models
```
TODO
```

## Experiment Data

The original paper uses **NaturalQuestions-Open**, a dataset containing historical queries submitted to Google's search engine, paired with human-annotated answers from Wikipedia. Specifically, they selected 2,655 queries where the annotated long answer is in paragraph form rather than a list or table. For their experiments, they utilized passages from Wikipedia (chunks of up to 100 tokens) as documents within the input contexts.

For each query, the dataset includes one document that contains the correct answer and *k − 1* distractor documents.
* The answer document is a Wikipedia paragraph sourced from the NaturalQuestions annotations.
* The *k − 1* distractors are retrieved using a Contriever model fine-tuned on MS-MARCO. These are topically relevant Wikipedia chunks that do not include the answer.

The `lost-in-the-middle` team included the exact dataset used during experiments in their Git repository. This includes three folders containing 10, 20, and 30 document JSON files, with the following fields:
* question (str): the prompt's question (query).
* answers (List[str]): a list of short answers.
* ctxs (List[Dict[str, Any]]): a list of documents used for the prompt’s context.
  * Important fields: `title`, `text`, `hasanswer`, and `isgold`.
  * We used the `lost-in-the-middle` original `Document` object "as is" (see `common/entities.py `- `Document`).
* nq_annotated_gold (Dict[str, Any]): contains the original annotated answers data.
  * Important fields: full chunk with answer, long answers, short answers, etc.
  * The `answers` field used in the original paper corresponds to the `short answers` from the `nq_annotated_gold` field.


## Experiment Description

Originally, the `lost-in-the-middle` paper examined two tasks: "multi-document question answering and key-value retrieval".
We focused on the "multi-document question answering" task and experimeneted two of its variants:

* **Gold Index Change:**
  In this variant we take a list of 10/20/30 documents, and change the location of the *gold* (correct) answer. For example, with 10 documents we examine the indicies of 0, 4 and 9. This experiment aims measure how the position of the correct answer affects the final output of the LLM.

* **Number of Documents Change:**
  In this variant, we fix the *gold* index (e.g., 0, 4, or 9) and vary the total number of documents: 10, 20, and 30. For example, if using gold index 0, we collect all document lists of sizes 10, 20, and 30 where the gold answer appears at index 0. Indices like 14 or 19 are excluded, as they don’t exist in the 10-document setting.

We implemented each variant as a separate class (inheriting from a shared abstract base class for common functionality) and created a runner class that dynamically identifies the target experiment and executes it accordingly.

We used `vLLM` to manage LLM / SSM operations over GPU and HuggingFace `transformers` to dynamically add the model-specific instruction tokens to the prompt.

We used `poetry` as a dependency manager and `pyest` to test our implementations. 

## Evaluation
```
TODO
```

## Results
```
TODO
```

## Future Work
```
TODO
```

## References

This project builds upon the findings from the paper `Lost in the Middle: How Language Models Use Long Contexts`. We extend this work by applying similar methodologies to State Space Models (SSMs), investigating how they handle large-context inputs compared to Large Language Models (LLMs).

Portions of this code are based on the original implementation from Lost in the Middle, licensed under the MIT License. Additional contributions have been made by Gal Sarid (2025).

For more details on the original study, see:

```
@misc{liu-etal:2023:arxiv,
  author    = {Nelson F. Liu  and  Kevin Lin  and  John Hewitt  and Ashwin Paranjape  and Michele Bevilacqua  and  Fabio Petroni  and  Percy Liang},
  title     = {Lost in the Middle: How Language Models Use Long Contexts},
  note      = {arXiv:2307.03172},
  year      = {2023}
}
```