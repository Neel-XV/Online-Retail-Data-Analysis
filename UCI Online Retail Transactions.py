# Databricks notebook source
# DBTITLE 1,Install UCI ML Repo Package and Restart Python
# MAGIC %pip install ucimlrepo --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Initialize Spark Session and Import Libraries
# Import required libraries
from ucimlrepo import fetch_ucirepo 
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
import datetime

# Initialize Spark session
spark = SparkSession.builder.appName("OnlineRetailAnalytics").getOrCreate()

# COMMAND ----------

# DBTITLE 1,Ingest Online Retail Data from UCI to Spark DataFrame
# ==========================================
# BRONZE LAYER - DATA INGESTION
# ==========================================

print("=== BRONZE LAYER: DATA INGESTION ===")

# Fetch the Online Retail dataset from UCI repository
online_retail = fetch_ucirepo(id=352)

# Define explicit schema for data consistency
schema = StructType([
    StructField("InvoiceNo", StringType(), True),
    StructField("StockCode", StringType(), True),
    StructField("Description", StringType(), True),
    StructField("Quantity", IntegerType(), True),
    StructField("InvoiceDate", StringType(), True),
    StructField("UnitPrice", DoubleType(), True),
    StructField("CustomerID", DoubleType(), True),
    StructField("Country", StringType(), True)
])

# Convert pandas DataFrame to Spark DataFrame with schema
retail_spark_df = spark.createDataFrame(online_retail.data.original, schema=schema)

print(f"Total records ingested: {retail_spark_df.count():,}")
retail_spark_df.printSchema()

# COMMAND ----------

# DBTITLE 1,Clean and Transform Retail Data in Bronze Layer
# ==========================================
# BRONZE LAYER - BASIC DATA CLEANING
# ==========================================

print("=== BRONZE LAYER: BASIC CLEANING ===")

# Apply basic transformations and filters
retail_clean_df = retail_spark_df \
    .withColumn("InvoiceDate", to_timestamp(col("InvoiceDate"), "M/d/yyyy H:mm")) \
    .withColumn("CustomerID", col("CustomerID").cast("integer")) \
    .withColumn("TotalPrice", col("Quantity") * col("UnitPrice")) \
    .filter(col("Quantity") > 0) \
    .filter(col("UnitPrice") > 0) \
    .filter(col("CustomerID").isNotNull())

# Display data quality metrics
print(f"Records after cleaning: {retail_clean_df.count():,}")
print(f"Unique customers: {retail_clean_df.select('CustomerID').distinct().count():,}")

# Show date range of the dataset
date_range = retail_clean_df.agg(min('InvoiceDate'), max('InvoiceDate')).collect()[0]
print(f"Date range: {date_range[0]} to {date_range[1]}")

# Save Bronze layer to Delta table
retail_clean_df.write \
    .format("delta") \
    .mode("overwrite") \
    .option("mergeSchema", "true") \
    .saveAsTable("online_retail_bronze")

print(" Bronze layer saved: online_retail_bronze")

# COMMAND ----------

# DBTITLE 1,Enrich and Categorize Data in Silver Layer
# ==========================================
# SILVER LAYER - DATA ENRICHMENT
# ==========================================

print("=== SILVER LAYER: DATA ENRICHMENT ===")

# Read from Bronze layer
bronze_df = spark.table("online_retail_bronze")

# Add time-based features and business categorizations
silver_df = bronze_df \
    .withColumn("Year", year(col("InvoiceDate"))) \
    .withColumn("Month", month(col("InvoiceDate"))) \
    .withColumn("Quarter", quarter(col("InvoiceDate"))) \
    .withColumn("DayOfWeek", dayofweek(col("InvoiceDate"))) \
    .withColumn("Hour", hour(col("InvoiceDate"))) \
    .withColumn("TotalPrice", round(col("Quantity") * col("UnitPrice"), 2))

# Add day name for better readability
silver_df = silver_df.withColumn("DayName", 
    when(col("DayOfWeek") == 1, "Sunday")
    .when(col("DayOfWeek") == 2, "Monday")
    .when(col("DayOfWeek") == 3, "Tuesday")
    .when(col("DayOfWeek") == 4, "Wednesday")
    .when(col("DayOfWeek") == 5, "Thursday")
    .when(col("DayOfWeek") == 6, "Friday")
    .when(col("DayOfWeek") == 7, "Saturday")
)

# Add weekend indicator and product categorization
silver_df = silver_df \
    .withColumn("IsWeekend", when(col("DayOfWeek").isin([1, 7]), 1).otherwise(0)) \
    .withColumn("ProductCategory", 
        when(col("StockCode").startswith("POST"), "Postage")
        .when(col("StockCode").startswith("D"), "Discount")
        .when(col("StockCode").startswith("C"), "Cancelled")
        .when(col("StockCode").startswith("M"), "Manual")
        .when(col("StockCode").contains("BANK"), "Bank_Charges")
        .otherwise("Product")
    ) \
    .withColumn("created_at", current_timestamp())

print(f"Silver layer records: {silver_df.count():,}")

# COMMAND ----------

# DBTITLE 1,Deduplicate and Save Retail Data to Silver Layer
# Remove duplicate records based on business logic
print("Removing duplicate records...")

# Define window for deduplication
dedup_window = Window.partitionBy("InvoiceNo", "StockCode", "CustomerID").orderBy(col("InvoiceDate").desc())

# Keep only the latest record for each unique combination
silver_df = silver_df \
    .withColumn("row_num", row_number().over(dedup_window)) \
    .filter(col("row_num") == 1) \
    .drop("row_num")

# Save Silver layer with partitioning for better performance
silver_df.write \
    .format("delta") \
    .mode("overwrite") \
    .option("mergeSchema", "true") \
    .partitionBy("Year", "Month") \
    .saveAsTable("online_retail_silver")

print(f" Silver layer saved with {silver_df.count():,} records")

# COMMAND ----------

# DBTITLE 1,Analyze Customer RFM Metrics in Gold Layer
# ==========================================
# GOLD LAYER - CUSTOMER ANALYTICS (RFM)
# ==========================================

print("=== GOLD LAYER: CUSTOMER RFM ANALYSIS ===")

# Read from Silver layer
silver_df = spark.table("online_retail_silver")

# Get the latest date in the dataset for recency calculations
current_date = silver_df.agg(max("InvoiceDate")).collect()[0][0]
print(f"Analysis reference date: {current_date}")

# Calculate base RFM metrics for each customer
customer_base_metrics = silver_df \
    .filter(col("CustomerID").isNotNull()) \
    .filter(col("TotalPrice") > 0) \
    .groupBy("CustomerID", "Country") \
    .agg(
        # Recency: Days since last purchase
        datediff(lit(current_date), max("InvoiceDate")).alias("Recency"),
        # Frequency: Number of unique invoices (transactions)
        countDistinct("InvoiceNo").alias("Frequency"),
        # Monetary: Total amount spent
        round(sum("TotalPrice"), 2).alias("Monetary"),
        # Additional customer insights
        count("*").alias("TotalItems"),
        countDistinct("StockCode").alias("UniqueProducts"),
        min("InvoiceDate").alias("FirstPurchase"),
        max("InvoiceDate").alias("LastPurchase"),
        round(avg("TotalPrice"), 2).alias("AvgOrderValue"),
        round(sum("TotalPrice") / countDistinct("InvoiceNo"), 2).alias("AvgBasketValue")
    )

print(f"Total customers analyzed: {customer_base_metrics.count():,}")

# COMMAND ----------

# DBTITLE 1,Validate RFM Data Quality Before Scoring
# Perform data quality validation before RFM scoring
print("=== RFM DATA QUALITY VALIDATION ===")

# Check for invalid data that could skew RFM analysis
invalid_monetary = customer_base_metrics.filter(col("Monetary") <= 0).count()
invalid_frequency = customer_base_metrics.filter(col("Frequency") <= 0).count()
invalid_recency = customer_base_metrics.filter(col("Recency") < 0).count()

print(f"Customers with invalid monetary values: {invalid_monetary}")
print(f"Customers with invalid frequency values: {invalid_frequency}")
print(f"Customers with invalid recency values: {invalid_recency}")

# Filter to clean dataset for accurate RFM scoring
clean_customers = customer_base_metrics.filter(
    (col("Monetary") > 0) & 
    (col("Frequency") > 0) & 
    (col("Recency") >= 0)
)

print(f"Clean customers for RFM scoring: {clean_customers.count():,}")

# COMMAND ----------

# DBTITLE 1,Calculate and Validate RFM Scores Using Quintiles
# Calculate RFM scores using quintiles
print("Calculating RFM scores...")

# Apply quintile scoring for each RFM component
# Note: For Recency, lower values (more recent) get higher scores, so we reverse the ranking

# Define window specifications with explicit partitioning
# Using lit(1) to create a single partition for global quintiles across all customers
recency_window = Window.partitionBy(lit(1)).orderBy("Recency")
frequency_window = Window.partitionBy(lit(1)).orderBy(col("Frequency").desc())
monetary_window = Window.partitionBy(lit(1)).orderBy(col("Monetary").desc())

# Apply quintile scoring for each RFM component
# Note: For Recency, lower values (more recent) get higher scores, so we reverse the ranking
rfm_scored = clean_customers \
    .withColumn("R_Score", 6 - ntile(5).over(recency_window)) \
    .withColumn("F_Score", ntile(5).over(frequency_window)) \
    .withColumn("M_Score", ntile(5).over(monetary_window)) \
    .withColumn("RFM_Score", concat(col("R_Score"), col("F_Score"), col("M_Score")))

# Display RFM score distribution to validate proper quintile creation
print("RFM Score Distribution:")
print("R_Score (Recency):")
rfm_scored.groupBy("R_Score").count().orderBy("R_Score").show()
print("F_Score (Frequency):")
rfm_scored.groupBy("F_Score").count().orderBy("F_Score").show()
print("M_Score (Monetary):")
rfm_scored.groupBy("M_Score").count().orderBy("M_Score").show()

# COMMAND ----------

# DBTITLE 1,Segment Customers Based on RFM Score
# Apply customer segmentation based on RFM scores
print("Applying customer segmentation...")

customer_analytics_final = rfm_scored.withColumn("CustomerSegment", 
    # Champions: High value customers who buy frequently and recently
    when(col("RFM_Score").isin([
        "555", "554", "544", "545", "454", "455", "445", "553", "552"
    ]), "Champions")
    
    # Loyal Customers: Regular customers with good purchase history
    .when(col("RFM_Score").isin([
        "543", "444", "435", "355", "354", "345", "344", "335", "434", "343"
    ]), "Loyal_Customers")
    
    # Potential Loyalists: Recent customers with growth potential
    .when(col("RFM_Score").isin([
        "551", "541", "542", "533", "532", "531", "452", "451", "453", "342", "351", "352", "353"
    ]), "Potential_Loyalists")
    
    # New Customers: Recent first-time or low-frequency buyers
    .when(col("RFM_Score").isin([
        "512", "511", "421", "422", "412", "411", "311", "312", "313", "314", "321", "322"
    ]), "New_Customers")
    
    # At Risk: Good customers who haven't purchased recently
    .when(col("RFM_Score").isin([
        "155", "154", "144", "214", "215", "115", "114", "113", "145", "125", "124"
    ]), "At_Risk")
    
    # Cannot Lose: High-value customers at high risk of churning
    .when(col("RFM_Score").isin([
        "123", "122", "121", "223", "222", "221", "213", "231", "141", "142", "143", "112", "111"
    ]), "Cannot_Lose")
    
    # Promising: New customers with decent purchase potential
    .when(col("RFM_Score").isin([
        "244", "245", "254", "255", "334", "325", "324", "323"
    ]), "Promising")
    
    # Need Attention: Below average customers who need activation
    .when(col("RFM_Score").isin([
        "253", "252", "251", "243", "242", "241", "235", "234", "233", "232"
    ]), "Need_Attention")
    
    # All other combinations
    .otherwise("Others")
) \
.withColumn("created_at", current_timestamp())

# Display customer segment distribution
print("Customer Segment Distribution:")
customer_analytics_final.groupBy("CustomerSegment").count().orderBy(col("count").desc()).show()

# COMMAND ----------

# DBTITLE 1,Save and Display Customer Analytics in Gold Layer
# Save customer analytics to Gold layer
customer_analytics_final.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("gold_customer_analytics")

print(" Customer Analytics saved to: gold_customer_analytics")

# Display sample results for verification
print("Sample Customer Analytics Results:")
customer_analytics_final.select(
    "CustomerID", "Country", "Recency", "Frequency", "Monetary", 
    "R_Score", "F_Score", "M_Score", "RFM_Score", "CustomerSegment"
).show(10)

# COMMAND ----------

# DBTITLE 1,Analyze and Save Product Performance in Gold Layer
# ==========================================
# GOLD LAYER - PRODUCT PERFORMANCE ANALYTICS
# ==========================================

print("=== GOLD LAYER: PRODUCT PERFORMANCE ===")

# Analyze product performance metrics focusing on actual products
product_performance = silver_df \
    .filter(col("ProductCategory") == "Product") \
    .groupBy("StockCode", "Description", "ProductCategory") \
    .agg(
        # Sales volume metrics
        sum("Quantity").alias("TotalQuantitySold"),
        round(sum("TotalPrice"), 2).alias("TotalRevenue"),
        
        # Customer engagement metrics
        countDistinct("CustomerID").alias("UniqueCustomers"),
        countDistinct("InvoiceNo").alias("UniqueOrders"),
        count("*").alias("TotalTransactions"),
        
        # Average performance metrics
        round(avg("UnitPrice"), 2).alias("AvgUnitPrice"),
        round(avg("Quantity"), 2).alias("AvgQuantityPerOrder"),
        round(sum("TotalPrice") / countDistinct("InvoiceNo"), 2).alias("AvgRevenuePerOrder")
    )

# Define window specifications for global rankings
revenue_window = Window.partitionBy(lit(1)).orderBy(col("TotalRevenue").desc())
quantity_window = Window.partitionBy(lit(1)).orderBy(col("TotalQuantitySold").desc())
popularity_window = Window.partitionBy(lit(1)).orderBy(col("UniqueCustomers").desc())

# Add ranking metrics for easy identification of top performers
product_performance = product_performance \
    .withColumn("RevenueRank", dense_rank().over(revenue_window)) \
    .withColumn("QuantityRank", dense_rank().over(quantity_window)) \
    .withColumn("PopularityRank", dense_rank().over(popularity_window)) \
    .withColumn("created_at", current_timestamp())

# Save product performance analytics
product_performance.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("gold_product_performance")

print(" Product Performance saved to: gold_product_performance")
print(f"Products analyzed: {product_performance.count():,}")

# Show top performing products
print("Top 10 Products by Revenue:")
product_performance.select(
    "StockCode", "Description", "TotalRevenue", "TotalQuantitySold", "UniqueCustomers"
).orderBy(col("TotalRevenue").desc()).show(10, truncate=False)

# COMMAND ----------

# DBTITLE 1,Create and Save Sales Time Series in Gold Layer
# ==========================================
# GOLD LAYER - TIME SERIES SALES ANALYTICS
# ==========================================

print("=== GOLD LAYER: SALES TIME SERIES ===")

# Create comprehensive time-based sales analytics
sales_timeseries = silver_df.groupBy("Year", "Month", "Quarter", "DayName", "IsWeekend", "Hour") \
    .agg(
        # Revenue and volume metrics
        round(sum("TotalPrice"), 2).alias("TotalRevenue"),
        sum("Quantity").alias("TotalQuantity"),
        
        # Transaction metrics
        countDistinct("InvoiceNo").alias("TotalOrders"),
        countDistinct("CustomerID").alias("UniqueCustomers"),
        countDistinct("StockCode").alias("UniqueProducts"),
        
        # Average performance metrics
        round(avg("TotalPrice"), 2).alias("AvgOrderValue"),
        round(sum("TotalPrice") / countDistinct("InvoiceNo"), 2).alias("AvgBasketValue")
    ) \
    .withColumn("created_at", current_timestamp())

growth_window = Window.partitionBy(lit(1)).orderBy("Year", "Month", "Quarter", "Hour")
sales_timeseries = sales_timeseries.withColumn(
    "RevenueGrowth", 
    col("TotalRevenue") - lag("TotalRevenue", 1).over(growth_window)
)

# Save time series analytics with partitioning for better query performance
sales_timeseries.write \
    .format("delta") \
    .mode("overwrite") \
    .partitionBy("Year") \
    .saveAsTable("gold_sales_timeseries")

print(" Sales Time Series saved to: gold_sales_timeseries")

# Show monthly trends
print("Monthly Sales Trends:")
monthly_summary = sales_timeseries.groupBy("Year", "Month") \
    .agg(
        sum("TotalRevenue").alias("MonthlyRevenue"),
        sum("TotalOrders").alias("MonthlyOrders")
    ).orderBy("Year", "Month")

monthly_summary.show(12)

# COMMAND ----------

# DBTITLE 1,Analyze and Save Geographic Performance in Gold Layer
# ==========================================
# GOLD LAYER - GEOGRAPHIC PERFORMANCE
# ==========================================

print("=== GOLD LAYER: GEOGRAPHIC PERFORMANCE ===")

# Analyze performance by country and time period
geographic_performance = silver_df.groupBy("Country", "Year", "Month") \
    .agg(
        # Financial metrics
        round(sum("TotalPrice"), 2).alias("TotalRevenue"),
        sum("Quantity").alias("TotalQuantity"),
        
        # Customer metrics
        countDistinct("CustomerID").alias("UniqueCustomers"),
        countDistinct("InvoiceNo").alias("TotalOrders"),
        countDistinct("StockCode").alias("UniqueProducts"),
        round(avg("TotalPrice"), 2).alias("AvgOrderValue")
    )

# Add country rankings within each time period
geographic_performance = geographic_performance \
    .withColumn("RevenueRank", 
               dense_rank().over(Window.partitionBy("Year", "Month").orderBy(col("TotalRevenue").desc()))) \
    .withColumn("CustomerRank", 
               dense_rank().over(Window.partitionBy("Year", "Month").orderBy(col("UniqueCustomers").desc()))) \
    .withColumn("created_at", current_timestamp())

# Save geographic performance with partitioning
geographic_performance.write \
    .format("delta") \
    .mode("overwrite") \
    .partitionBy("Year") \
    .saveAsTable("gold_geographic_performance")

print(" Geographic Performance saved to: gold_geographic_performance")

# Show top countries by revenue
print("Top Countries by Total Revenue:")
country_totals = geographic_performance.groupBy("Country") \
    .agg(
        round(sum("TotalRevenue"), 2).alias("TotalRevenue"),
        sum("UniqueCustomers").alias("TotalCustomers")
    ).orderBy(col("TotalRevenue").desc())

country_totals.show(10)

# COMMAND ----------

# DBTITLE 1,Aggregate and Save Daily Business Metrics in Gold Layer
# ==========================================
# GOLD LAYER - DAILY BUSINESS METRICS
# ==========================================

print("=== GOLD LAYER: DAILY BUSINESS METRICS ===")

# Create daily aggregated metrics for operational dashboards
daily_metrics = silver_df.groupBy(to_date("InvoiceDate").alias("Date")) \
    .agg(
        # Daily performance metrics
        round(sum("TotalPrice"), 2).alias("DailyRevenue"),
        sum("Quantity").alias("DailyQuantity"),
        countDistinct("InvoiceNo").alias("DailyOrders"),
        countDistinct("CustomerID").alias("DailyCustomers"),
        countDistinct("StockCode").alias("DailyProducts"),
        
        # Daily averages
        round(avg("TotalPrice"), 2).alias("AvgOrderValue"),
        round(sum("TotalPrice") / countDistinct("CustomerID"), 2).alias("RevenuePerCustomer")
    )

# Add time dimensions for analysis
daily_metrics = daily_metrics \
    .withColumn("Year", year("Date")) \
    .withColumn("Month", month("Date")) \
    .withColumn("DayOfWeek", dayofweek("Date")) \
    .withColumn("IsWeekend", when(col("DayOfWeek").isin([1, 7]), 1).otherwise(0))

# Calculate moving averages for trend analysis
daily_window_7 = Window.partitionBy(lit(1)).orderBy("Date").rowsBetween(-6, 0)
daily_window_30 = Window.partitionBy(lit(1)).orderBy("Date").rowsBetween(-29, 0)

daily_metrics = daily_metrics \
    .withColumn("MovingAvg7Days", round(avg("DailyRevenue").over(daily_window_7), 2)) \
    .withColumn("MovingAvg30Days", round(avg("DailyRevenue").over(daily_window_30), 2)) \
    .withColumn("created_at", current_timestamp())

# Save daily metrics with partitioning for efficient queries
daily_metrics.write \
    .format("delta") \
    .mode("overwrite") \
    .partitionBy("Year", "Month") \
    .saveAsTable("gold_daily_metrics")

print(" Daily Metrics saved to: gold_daily_metrics")
print(f"Daily records created: {daily_metrics.count():,}")

# Show recent daily performance
print("Recent Daily Performance (Last 10 Days):")
daily_metrics.select(
    "Date", "DailyRevenue", "DailyOrders", "DailyCustomers", "MovingAvg7Days"
).orderBy(col("Date").desc()).show(10)

# COMMAND ----------

# DBTITLE 1,Summarize Business Insights and Segmentation Analysis
# ==========================================
# BUSINESS INSIGHTS SUMMARY
# ==========================================

print("=== COMPREHENSIVE BUSINESS INSIGHTS ===")

# Executive summary metrics
exec_summary = spark.sql("""
    SELECT 
        ROUND(SUM(Monetary), 2) as TotalRevenue,
        COUNT(*) as TotalCustomers,
        ROUND(AVG(Monetary), 2) as AvgCustomerValue,
        ROUND(AVG(Frequency), 1) as AvgPurchaseFrequency,
        ROUND(AVG(Recency), 1) as AvgDaysSinceLastPurchase
    FROM gold_customer_analytics
""")

print("Executive Summary:")
exec_summary.show()

# Customer segmentation business insights
segment_insights = spark.sql("""
    SELECT 
        CustomerSegment,
        COUNT(*) as CustomerCount,
        ROUND(AVG(Monetary), 2) as AvgSpending,
        ROUND(AVG(Frequency), 1) as AvgFrequency,
        ROUND(AVG(Recency), 1) as AvgRecency,
        ROUND(SUM(Monetary), 2) as TotalRevenue,
        ROUND(SUM(Monetary) * 100.0 / (SELECT SUM(Monetary) FROM gold_customer_analytics), 2) as RevenueShare
    FROM gold_customer_analytics
    GROUP BY CustomerSegment
    ORDER BY TotalRevenue DESC
""")

print("Customer Segmentation Analysis:")
segment_insights.show()

# COMMAND ----------

print("=== DATA PIPELINE COMPLETED SUCCESSFULLY ===")
print("Gold Tables Created:")
print("   • gold_customer_analytics - RFM analysis and customer segmentation")
print("   • gold_product_performance - Product sales and popularity metrics") 
print("   • gold_sales_timeseries - Time-based sales analysis")
print("   • gold_geographic_performance - Country-wise performance")
print("   • gold_daily_metrics - Daily operational KPIs")

# COMMAND ----------

# MAGIC %md
# MAGIC <h2 style="text-align: center;">Visualizations</h2>

# COMMAND ----------

# DBTITLE 1,Customer Segment Count and Revenue Analysis
# Customer segment distribution
customer_segments_viz = spark.sql("""
    SELECT 
        CustomerSegment, 
        COUNT(*) as CustomerCount,
        ROUND(SUM(Monetary), 2) as TotalRevenue
    FROM gold_customer_analytics
    GROUP BY CustomerSegment
    ORDER BY CustomerCount DESC
""").toPandas()

print("Customer Segment Distribution:")
display(customer_segments_viz)

# COMMAND ----------

# DBTITLE 1,Monthly Revenue and Order Trends
# Monthly revenue and order trends
monthly_trends_viz = spark.sql("""
    SELECT 
        CONCAT(Year, '-', LPAD(Month, 2, '0')) as YearMonth,
        SUM(TotalRevenue) as MonthlyRevenue,
        SUM(TotalOrders) as MonthlyOrders
    FROM gold_sales_timeseries
    GROUP BY Year, Month
    ORDER BY Year, Month
""").toPandas()

print("Monthly Revenue Trends:")
display(monthly_trends_viz)

# COMMAND ----------

# DBTITLE 1,Top RFM Combos with Highest Customer Count and Revenue
# Top/Bottom performing RFM combinations
print("Top 10 RFM Combinations by Customer Count")
top_rfm_combos = spark.sql("""
    SELECT 
        CONCAT('R:', R_Score, ' F:', F_Score, ' M:', M_Score) as RFM_Combination,
        R_Score, F_Score, M_Score,
        CustomerCount,
        AvgSpending,
        ROUND(CustomerCount * AvgSpending, 2) as Revenue_Potential
    FROM (
        SELECT 
            R_Score, F_Score, M_Score,
            COUNT(*) as CustomerCount,
            ROUND(AVG(Monetary), 2) as AvgSpending
        FROM gold_customer_analytics
        GROUP BY R_Score, F_Score, M_Score
    ) rfm_data
    ORDER BY CustomerCount DESC
    LIMIT 10
""")

display(top_rfm_combos)

# COMMAND ----------

# DBTITLE 1,Customer Count vs Avg Spending
# Customer Count vs Average Spending (by Recency Score)

scatter_by_recency = spark.sql("""
    SELECT 
        R_Score,
        F_Score,
        M_Score,
        COUNT(*) as CustomerCount,
        ROUND(AVG(Monetary), 2) as AvgSpending,
        CONCAT('R:', R_Score, ' F:', F_Score, ' M:', M_Score) as RFM_Label
    FROM gold_customer_analytics
    GROUP BY R_Score, F_Score, M_Score
    ORDER BY R_Score, F_Score, M_Score
""")
print("Customer Count vs Average Spending by R_Score")
display(scatter_by_recency)

# COMMAND ----------

# DBTITLE 1,Top 15 Products by Sales Performance
# Top products by revenue
top_products_viz = spark.sql("""
    SELECT 
        CASE 
            WHEN LENGTH(Description) > 30 THEN CONCAT(SUBSTRING(Description, 1, 30), '...')
            ELSE Description 
        END as ProductName,
        TotalRevenue,
        TotalQuantitySold,
        UniqueCustomers
    FROM gold_product_performance
    ORDER BY TotalRevenue DESC
    LIMIT 15
""").toPandas()
print("Top 15 Products by Revenue")
display(top_products_viz)