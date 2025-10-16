using Microsoft.ML.Data;

namespace NetflixAnalysisApp.MLModel
{
    public class TitlePrediction
    {
        [ColumnName("PredictedLabel")] 
        public string PredictedType;
        public float[] Score { get; set; }

    }
}
