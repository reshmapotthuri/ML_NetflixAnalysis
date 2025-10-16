using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;
using Microsoft.ML;
using NetflixAnalysisApp.MLModel;
using System;
using System.Data;
using System.Globalization;
using System.Reflection;
using System.Text.RegularExpressions;
using CsvHelper;
using CsvHelper.Configuration;
using CsvHelper.Configuration.Attributes;

namespace NetflixAnalysisApp.Pages
{
    public class IndexModel : PageModel
    {
        private readonly ILogger<IndexModel> _logger;
        private static readonly string modelPath = "Model.zip";
        private PredictionEngine<TitleData, TitlePrediction> predictionEngine;
        public List<TypeCount> TypeCounts { get; set; }
        public List<YearCount> YearCounts { get; set; }
        public List<CountryCount> TopCountries { get; set; }
        public List<GenreCount> TopGenres { get; set; }




        public IndexModel(ILogger<IndexModel> logger)
        {
            _logger = logger;
        }

        public void OnGet()
        {
            var mlContext = new MLContext();
            var modelPath = Path.Combine(Directory.GetCurrentDirectory(), "wwwroot", "models", "Model.zip");
            var dataPath = Path.Combine(Directory.GetCurrentDirectory(), "wwwroot", "models", "netflix_titles.csv");
            IDataView dataView = mlContext.Data.LoadFromTextFile<TitleData>(dataPath, separatorChar: ',', hasHeader: true);
            ;
            var dataEnumerable = mlContext.Data.CreateEnumerable<TitleData>(
                                    dataView,
                          reuseRowObject: false
                            ).Take(100).ToList();
            List<TitleData> records;
            using (var reader = new StreamReader(dataPath))           
            using (var csv = new CsvReader(reader, CultureInfo.InvariantCulture))
            {
                csv.Context.RegisterClassMap<TitleDataMap>();
                records = csv.GetRecords<TitleData>().ToList();

            }
             var model = mlContext.Model.Load(modelPath, out var schema);
            var predictor = mlContext.Model.CreatePredictionEngine<TitleData, TitlePrediction>(model);

            TypeCounts = records
             .GroupBy(p => p.Type)
             .Select(g => new TypeCount
             {
                 Type = g.Key,
                 Count = g.Count()
             })
             .ToList();

            YearCounts = records
                .Select(r => new
                {
                    ParsedYear = int.TryParse(r.Year, out var y) ? y : 0
                })
                .Where(x => x.ParsedYear > 1900)
                .GroupBy(x => x.ParsedYear)
                .Select(g => new YearCount
                {
                    Year = g.Key,
                    Count = g.Count()
                })
                .OrderBy(y => y.Year)
                .ToList();

            TopCountries = records
           .Where(r => !string.IsNullOrWhiteSpace(r.Country))
           .SelectMany(r => r.Country.Split(','))
           .Select(c => c.Trim())
           .GroupBy(c => c)
           .Select(g => new CountryCount
           {
               Country = g.Key,
               Count = g.Count()
           })
           .OrderByDescending(c => c.Count)
           .Take(10)
           .ToList();

            TopGenres = records
           .Where(r => !string.IsNullOrWhiteSpace(r.Genre))
           .SelectMany(r => r.Genre.Split(','))
           .Select(g => g.Trim())
           .GroupBy(g => g)
           .Select(g => new GenreCount
           {
               Genre = g.Key,
               Count = g.Count()
           })
           .OrderByDescending(g => g.Count)
           .Take(10)
           .ToList();


        }     
    }

}
