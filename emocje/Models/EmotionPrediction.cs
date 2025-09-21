using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace emocje.Models
{
    public class EmotionPrediction
    {
        [ColumnName("PredictedEmotion")]
        public string PredictedEmotion { get; set; }
    }

}
