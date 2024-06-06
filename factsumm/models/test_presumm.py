# from extractive import ExtractiveSummarizer
from factsumm.models.extractive import ExtractiveSummarizer
import evaluate, json, os

model = ExtractiveSummarizer.load_from_checkpoint(
    "/mnt/colab_public/gebelangsn/pretrained_checkpoints/distilroberta-base-ext-sum/epoch3.ckpt"
)

test_set = []
for line in open(os.path.join("../data/Tweetsumm/", "test.jsonl"), "r").readlines():
    test_set.append(json.loads(line))

train_set = []
for line in open(os.path.join("../data/Tweetsumm/", "train.jsonl"), "r").readlines():
    train_set.append(json.loads(line))

# print("TEST SCORE")
# model.predict_sentences(
#     [
#         obj["dialog"].replace("\t", "").replace("\n", " [SEP] ").strip()
#         for obj in test_set
#     ]
# )
# print(
#     evaluate.load("rouge").compute(
#         predictions=[
#             model.predict(
#                 obj["dialog"].replace("\t", "").replace("\n", " [SEP] ").strip()
#             )
#             for obj in test_set
#         ],
#         references=[
#             obj["extractive_summaries"][0].replace("\t", "").replace("\n", " ").strip()
#             for obj in test_set
#         ],
#         rouge_types=["rouge1", "rouge2", "rougeL"],
#         use_stemmer=True,
#     )
# )

pseudos = [
    model.predict(obj["dialog"].replace("\t", "").replace("\n", " [SEP] ").strip())
    for obj in train_set[50:60]
]
for i in range(10):
    print("conv:")
    print(
        train_set[50 + i]["dialog"].replace("\t", "").replace("\n", " [SEP] ").strip()
    )
    print("pseudo:")
    print(pseudos[i])
    print("\nGT:")
    print(
        train_set[50 + i]["extractive_summaries"][0]
        .replace("\t", "")
        .replace("\n", " ")
        .strip()
    )
    print("=======")
print("PSEUDO-VALIDATION SCORE")
print(
    evaluate.load("rouge").compute(
        predictions=[
            model.predict(
                obj["dialog"].replace("\t", "").replace("\n", " [SEP] ").strip()
            )
            for obj in train_set[50:]
        ],
        references=[
            obj["extractive_summaries"][0].replace("\t", "").replace("\n", " ").strip()
            for obj in train_set[50:]
        ],
        rouge_types=["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )
)