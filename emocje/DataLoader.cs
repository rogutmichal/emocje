using emocje.Models;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace emocje
{
    public class DataLoader
    {
        public static List<TextData> LoadDataFromFile(string filePath)
        {
            var reviews = new List<TextData>();
            var lines = File.ReadAllLines(filePath);

            foreach (var line in lines)
            {
                var parts = line.Split(';');
                if (parts.Length == 2)
                {
                    var review = new TextData
                    {
                        Text = parts[0].Trim(),
                        Emotion = parts[1].Trim()
                    };


                    reviews.Add(review);
                }
            }
            return reviews;
        }
    }
}