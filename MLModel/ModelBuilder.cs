using Microsoft.ML;

namespace NetflixAnalysisApp.MLModel
{
    public class ModelBuilder
    {
        private static readonly string datapath = @"D:\Reshma-Projects\OCRMLClassifierApp\OCRMLClassifierApp\NetflixAnalysisApp\netflix_titles.csv";
        private static readonly string modelpath = "Model.zip";
        public static List<TypeCount> GetPredictedTypeCounts() 
        {
            var mlContext = new MLContext();

            //Load Data
            IDataView dataView = mlContext.Data.LoadFromTextFile<TitleData>(datapath, separatorChar: ',', hasHeader: true);
            ;
            var dataEnumerable = mlContext.Data.CreateEnumerable<TitleData>(
                                    dataView,
                          reuseRowObject: false
                            ).Take(100).ToList();

            // Clean nulls and empty strings
            foreach (var item in dataEnumerable)
            {
                item.Title = string.IsNullOrWhiteSpace(item.Title) ? "Unknown" : item.Title;
                item.Genre = string.IsNullOrWhiteSpace(item.Genre) ? "Unknown" : item.Genre;
                item.Description = string.IsNullOrWhiteSpace(item.Description) ? "No description" : item.Description;
                item.Duration = string.IsNullOrWhiteSpace(item.Duration) ? "Unknown" : item.Duration;
                item.Type = string.IsNullOrWhiteSpace(item.Type) ? "Movie" : item.Type; // default label
            }

            // Reload cleaned data
            var smallData = mlContext.Data.LoadFromEnumerable(dataEnumerable);

            var pipeLine = BuildPipeline(mlContext);

            var model = pipeLine.Fit(smallData);

            var predictions = model.Transform(dataView);
            // var metrics = mlContext.MulticlassClassification.Evaluate(predictions);


            //save Model
            mlContext.Model.Save(model, dataView.Schema, modelpath);
            Console.WriteLine("Model trained and saved succesfuly");

            var predictor = mlContext.Model.CreatePredictionEngine<TitleData, TitlePrediction>(model);
            var result = predictor.Predict(new TitleData
            {
                Title = "Stranger Things",
                Genre = "Sci-Fi",
                Description = "A group of kids uncover supernatural mysteries in their town.",
                Duration = "4 Seasons"
            });
            Console.WriteLine($"Predicted Type: {result.PredictedType}");
           
            var predictedResults = mlContext.Data.CreateEnumerable<TitlePrediction>(
                predictions, reuseRowObject: false);

            var predictedGroups = dataEnumerable
                .Where(x => !string.IsNullOrWhiteSpace(x.Title) &&
                !string.IsNullOrWhiteSpace(x.Type))
               .Select(x => predictor.Predict(x))
               .GroupBy(p => p.PredictedType)
               .Select(g => new TypeCount
               {
                   Type = g.Key,                  
                   Count = g.Count()
               })
               .ToList();

            return predictedGroups;

        }
        public static IEstimator<ITransformer> BuildPipeline(MLContext mlContext)
        {
         var pipeline = mlContext.Transforms.Text.FeaturizeText("TitleFeats", nameof(TitleData.Title))
                .Append(mlContext.Transforms.Text.FeaturizeText("GenreFeats", nameof(TitleData.Genre)))
                .Append(mlContext.Transforms.Text.FeaturizeText("DescFeats", nameof(TitleData.Description)))
                .Append(mlContext.Transforms.Text.FeaturizeText("DurationFeats", nameof(TitleData.Duration)))
                .Append(mlContext.Transforms.Concatenate("Features", "TitleFeats", "GenreFeats", "DescFeats", "DurationFeats"))
                .Append(mlContext.Transforms.Conversion.MapValueToKey("Label", nameof(Type)))
                .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
            return pipeline;
        }
    }


}
