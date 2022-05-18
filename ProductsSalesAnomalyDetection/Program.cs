using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.Data.SqlClient;
namespace ProductsSalesAnomalyDetection
{
    class Program
    {
        //static string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "product-sales.csv");
        //assign the Number of records in dataset file to constant variable
        const int _docsize = 36;

        [Obsolete]
        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();
            DatabaseLoader loader = mlContext.Data.CreateDatabaseLoader<ProductSalesData>();
             string connectionString = @"Data Source=LSTKBA228336\\SQLEXPRESS;Database=SalesProducts;Integrated Security=True;Connect Timeout=30";
            string sqlCommand = "SELECT  Month, ProductSales FROM ProductSales";
            var datosBD = ListTablesFromDb("", "", "", "");
            DatabaseSource dbSource = new DatabaseSource(System.Data.SqlClient.SqlClientFactory.Instance, connectionString, sqlCommand);
            IDataView dataView = loader.Load(dbSource);

           // IDataView dataView = mlContext.Data.LoadFromTextFile<ProductSalesData>(path: _dataPath, hasHeader: true, separatorChar: ',');
            DetectSpike(mlContext, _docsize, dataView);
        }
        static IDataView CreateEmptyDataView(MLContext mlContext)
        {
            // Create empty DataView. We just need the schema to call Fit() for the time series transforms
            IEnumerable<ProductSalesData> enumerableData = new List<ProductSalesData>();
            return mlContext.Data.LoadFromEnumerable(enumerableData);
        }

        [Obsolete]
        static void DetectSpike(MLContext mlContext, int docSize, IDataView productSales)
        {
            var iidSpikeEstimator = mlContext.Transforms.DetectIidSpike(outputColumnName: nameof(ProductSalesPrediction.Prediction), 
            inputColumnName: nameof(ProductSalesData.ProductSales), confidence: 95, pvalueHistoryLength: docSize / 4);
            ITransformer iidSpikeTransform = iidSpikeEstimator.Fit(CreateEmptyDataView(mlContext));
            IDataView transformedData = iidSpikeTransform.Transform(productSales);
            var predictions = mlContext.Data.CreateEnumerable<ProductSalesPrediction>(transformedData, reuseRowObject: false);
            Console.WriteLine("Alert\tScore\tP-Value");
            foreach (var p in predictions)
            {
                Console.WriteLine(System.Text.Json.JsonSerializer.Serialize(p));
                var results = $"{p.Prediction[0]}\t{p.Prediction[1]:f2}\t{p.Prediction[2]:F2}";

                if (p.Prediction[0] == 1)
                {
                    results += " <-- Spike detected";
                }

                Console.WriteLine(results);
            }
            Console.WriteLine("");

        }
    
        static List<string> ListTablesFromDb(string server, string database, string user, string password)
        {
            string connectionString = string.Empty;
            server = "LSTKBA228336\\SQLEXPRESS";
            database = "SalesProducts";
            //if (String.IsNullOrEmpty(user) && String.IsNullOrEmpty(password))
            //{
                connectionString =
                $"Data Source={server};" +
                $"Initial Catalog={database};" +
                $"Integrated Security=SSPI;";
            //}
            //else
            //{
            //    connectionString = $"Persist Security Info=False;User ID={user};Password={password};Initial Catalog={database};Server={server}";
            //}
            //string queryString =
            //$"SELECT TABLE_NAME FROM {database}.INFORMATION_SCHEMA.TABLES  WHERE TABLE_TYPE = 'BASE TABLE'";
            string queryString =
            $"SELECT * FROM Product_Sales";


            using (SqlConnection connection = new SqlConnection(connectionString))
            {

                SqlCommand command = new SqlCommand(queryString, connection);

                try
                {
                    connection.Open();
                    SqlDataReader reader = command.ExecuteReader();

                    List<string> tables = new List<string>();
                    while (reader.Read())
                    {
                        tables.Add(Convert.ToString(reader["ProductSales"]));
                    }
                    reader.Close();
                    return tables;
                }
                catch (Exception)
                {
                    List<string> tables = new List<string>();
                    return tables;
                }

            }

        }
    }

    public class ProductSalesData2
    {
        
        public string Month { get; set; }

        public float ProductSales { get; set; }
    }

}