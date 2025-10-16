using CsvHelper.Configuration;
using Microsoft.ML.Data;

namespace NetflixAnalysisApp.MLModel
{
    public class TitleData
    {
        [LoadColumn(0)]
        public string Title;
        [LoadColumn(1)]
        public string Genre;
        [LoadColumn(2)]
        public string Description;
        [LoadColumn(3)]
        public string Duration;
        [LoadColumn(4)]
        public string Type; // Label
        [LoadColumn(5)]
        public string Year;
        [LoadColumn(6)]
        public string Country;

    }
    public sealed class TitleDataMap : ClassMap<TitleData>
    {
        public TitleDataMap()
        {
            //Map(m => m.ShowId).Name("show_id");
            Map(m => m.Type).Name("type");
            Map(m => m.Title).Name("title");
            //Map(m => m.Director).Name("director");
            //Map(m => m.Cast).Name("cast");
            Map(m => m.Country).Name("country");
            //Map(m => m.DateAdded).Name("date_added");
            Map(m => m.Year).Name("release_year");
            //Map(m => m.Rating).Name("rating");
            Map(m => m.Duration).Name("duration");
            Map(m => m.Genre).Name("listed_in");
            Map(m => m.Description).Name("description");
        }
    }
}
