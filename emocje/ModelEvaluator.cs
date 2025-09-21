using emocje.Models;
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace emocje
{
    public class ModelEvaluator
    {
        public static void TestModel(MLContext context, ITransformer model, List<TextData> reviews, string datasetName)
        {
            var testData = context.Data.LoadFromEnumerable(reviews);
            var labelMapping = context.Transforms.Conversion.MapValueToKey("Label", "Emotion");
            var testDataWithLabel = labelMapping.Fit(testData).Transform(testData);

            var predictions = model.Transform(testDataWithLabel);
            var metrics = context.MulticlassClassification.Evaluate(predictions, labelColumnName: "Label");

            Console.WriteLine($"\n=== METRYKI MODELU – {datasetName.ToUpper()} ===");
            Console.WriteLine($"🎯 Accuracy (Micro): {metrics.MicroAccuracy:P2}");
            Console.WriteLine($"📊 Accuracy (Macro): {metrics.MacroAccuracy:P2}");
            Console.WriteLine($"📉 LogLoss: {metrics.LogLoss:F4}");
            Console.WriteLine("======================");

            var predictionEngine = context.Model.CreatePredictionEngine<TextData, EmotionPrediction>(model);

            var labelStats = new Dictionary<string, (int correct, int total)>();
            int correctGlobal = 0;

            int misclassifiedJoyAsLoveCount = 0;
            var misclassifiedJoyAsLoveExamples = new List<string>();

            foreach (var review in reviews)
            {
                var predicted = predictionEngine.Predict(review).PredictedEmotion;
                var actual = review.Emotion;

                if (!labelStats.ContainsKey(actual))
                    labelStats[actual] = (0, 0);

                var current = labelStats[actual];
                labelStats[actual] = (
                    correct: current.correct + (predicted == actual ? 1 : 0),
                    total: current.total + 1
                );

                if (predicted == actual) correctGlobal++;

                if (actual == "joy" && predicted == "love")
                {
                    misclassifiedJoyAsLoveCount++;
                    misclassifiedJoyAsLoveExamples.Add(review.Text);
                }
            }

            double simpleAccuracy = (double)correctGlobal / reviews.Count;
            Console.WriteLine($"🧪 Ręcznie liczona skuteczność ({datasetName}): {simpleAccuracy:P2} ({correctGlobal}/{reviews.Count})");

            Console.WriteLine($"\n📈 Skuteczność dla każdej emocji ({datasetName}):");
            foreach (var kvp in labelStats.OrderBy(k => k.Key))
            {
                var emotion = kvp.Key;
                var correct = kvp.Value.correct;
                var total = kvp.Value.total;
                double acc = (double)correct / total;
                Console.WriteLine($"  - {emotion}: {acc:P2} ({correct}/{total})");
            }

            var confusionMatrix = metrics.ConfusionMatrix;

            var labelBuffer = new VBuffer<ReadOnlyMemory<char>>();
            predictions.Schema["PredictedLabel"].GetKeyValues(ref labelBuffer);
            var labels = labelBuffer.DenseValues().Select(v => v.ToString()).ToList();

            Console.WriteLine($"\n📌 Confusion Matrix ({datasetName}):");
            int columnWidth = 10;
            Console.Write($"{"Predicted →",-12}");
            foreach (var label in labels)
                Console.Write(string.Format("{0," + columnWidth + "}", label));
            Console.WriteLine();
            for (int i = 0; i < confusionMatrix.NumberOfClasses; i++)
            {
                Console.Write($"{labels[i],-12}");
                for (int j = 0; j < confusionMatrix.NumberOfClasses; j++)
                {
                    Console.Write(string.Format("{0," + columnWidth + "}", confusionMatrix.Counts[i][j]));
                }
                Console.WriteLine();
            }

            Console.WriteLine($"\n🎯 Per-class metrics ({datasetName}):");
            Console.WriteLine($"{"Label",-12} {"Precision",10} {"Recall",10} {"F1-Score",10}");
            for (int i = 0; i < confusionMatrix.NumberOfClasses; i++)
            {
                string label = labels[i];
                double precision = confusionMatrix.PerClassPrecision[i];
                double recall = confusionMatrix.PerClassRecall[i];
                double f1 = 2 * (precision * recall) / (precision + recall + 1e-10);
                Console.WriteLine($"{label,-12} {precision,10:P2} {recall,10:P2} {f1,10:P2}");
            }

            int totalSamples = reviews.Count;
            double weightedSum = 0.0;

            foreach (var kvp in labelStats)
            {
                var correct = kvp.Value.correct;
                var total = kvp.Value.total;
                double classAcc = (double)correct / total;
                double weight = (double)total / totalSamples;
                weightedSum += classAcc * weight;
            }

            Console.WriteLine($"\n⚖️ Weighted Accuracy: {weightedSum:P2}");


        }

    }
}
