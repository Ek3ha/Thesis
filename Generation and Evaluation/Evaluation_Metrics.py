import pandas as pd
import numpy as np
import evaluate
from numpy import maximum as max

data = pd.read_csv(r"/data1/s3531643/thesis/Data/data4ektha.csv")
data_gen = pd.read_csv(r"/data1/s3531643/thesis/Code//data1/s3531643/thesis/Code/Generated_comments_FewShot1060_Diverse990.csv.csv")

data_gen = data_gen.dropna()

posts1 = data.groupby("text.x")["text.y"].groups
posts2 = data_gen.groupby("Post")["Comments"].groups



rouge = evaluate.load('rouge')
bleu = evaluate.load('bleu')
meteor = evaluate.load('meteor')
bertscore = evaluate.load('bertscore')

maximum_sores_rouge1 = []
maximum_sores_rouge2 = []
maximum_sores_rougeL = []
maximum_sores_rougesum = []

maximum_scores_bleu = []

maximum_scores_m = []
maximum_scores_bs = []


for i in posts2.keys():
    if i in posts1.keys():
        values1 = posts1[i]
        values2 = posts2[i]

        actual_values1 = data.loc[values1, 'text.y'].tolist()
        actual_values2 = data_gen.loc[values2, 'Comments'].tolist()

        max_rouge1 = 0.0
        max_rouge2 = 0.0
        max_rougeL = 0.0
        max_rougesum = 0.0
        max_bleuecore = 0.0
        max_m = 0.0
        max_bs = 0.0

        # Calculate ROUGE scores for all combinations and update max scores
        for j in actual_values1:
            for k in actual_values2:

                results = rouge.compute(predictions=[k], references=[j])
                print(max_rouge1,max_rouge2,results)
                max_rouge1 = max(max_rouge1, results["rouge1"])
                max_rouge2 = max(max_rouge2, results["rouge2"])
                max_rougeL = max(max_rougeL, results["rougeL"])
                max_rougesum = max(max_rougesum, results["rougeLsum"])

                try:
                    results_bleu = bleu.compute(predictions=[k], references=[j])
                    print(results_bleu)
                    max_bleuecore = max(max_bleuecore, results_bleu["bleu"])

                except ZeroDivisionError as e:
                    print("------------------ZERO ERROR---------------------------")
                    pass

                results_m = meteor.compute(predictions=[k], references=[j])
                print(results_m)
                max_m = max(max_m, results_m["meteor"])

                results_bs = bertscore.compute(predictions=[k], references=[j],lang ="nl")
                print(results_bs)
                max_bs = max(max_bs, results_bs["f1"][0])

        maximum_sores_rouge1.append(max_rouge1)
        maximum_sores_rouge2.append(max_rouge2)
        maximum_sores_rougeL.append(max_rougeL)
        maximum_sores_rougesum.append(max_rougesum)
        maximum_scores_bleu.append(max_bleuecore)
        maximum_scores_m.append(max_m)
        maximum_scores_bs.append(max_bs)



print(np.mean(maximum_sores_rouge1),np.mean(maximum_sores_rouge2),np.mean(maximum_sores_rougeL),np.mean(maximum_sores_rougesum),
        np.mean(maximum_scores_bleu),np.mean(maximum_scores_m),np.mean(maximum_scores_bs))


print(np.max(maximum_sores_rouge1),np.max(maximum_sores_rouge2),np.max(maximum_sores_rougeL),np.max(maximum_sores_rougesum),
        np.max(maximum_scores_bleu),np.max(maximum_scores_m),np.max(maximum_scores_bs))
