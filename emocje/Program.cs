using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;
using System.Linq;
using System.Text.RegularExpressions;
using emocje.Models;
using emocje;






public class EmotionBasedRecommendation
{
    private static readonly string ModelPath = "emotion_model.zip";
    private static readonly string TrainFilePath = "train.txt";
    private static readonly string ValFilePath = "val.txt";
    private static readonly string TestFilePath = "test.txt";

    public static void Main(string[] args)
    {
        var context = new MLContext();
        var trainTexts = DataLoader.LoadDataFromFile(TrainFilePath);

        ITransformer model;

        // DI initialization (assuming you've set it up)
        var trainer = new EmotionModelTrainer(context); // Tworzymy instancję EmotionModelTrainer

        if (!File.Exists(ModelPath))
        {
            Console.WriteLine("Training model...");
            model = trainer.TrainModel(trainTexts); // Teraz metoda 'TrainModel' działa na instancji
            context.Model.Save(model, context.Data.LoadFromEnumerable(trainTexts).Schema, ModelPath);
        }
        else
        {
            Console.WriteLine("Loading pre-trained model...");
            model = context.Model.Load(ModelPath, out _);
        }

        var exampleText = new TextData { Text = "I love the animation style!" };
        var top3Emotions = EmotionModel.PredictEmotion(context, model, exampleText);

        Console.WriteLine($"Review: {exampleText.Text}");
        Console.WriteLine("Top 3 emotion probabilities:");
        foreach (var kvp in top3Emotions)
        {
            Console.WriteLine($"{kvp.Key}: {kvp.Value:P2}");
        }

        var valReviews = DataLoader.LoadDataFromFile(ValFilePath);
        ModelEvaluator.TestModel(context, model, valReviews, "VALIDATION");

        var testReviews = DataLoader.LoadDataFromFile(TestFilePath);
        ModelEvaluator.TestModel(context, model, testReviews, "TEST");

        Console.ReadKey();
    }
}
