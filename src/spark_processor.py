"""
PySpark Structured Streaming Subprocessor for ARES.
Reads from a Kafka topic, applies transformations, and outputs to the console.
"""

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, when
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType

def create_spark_session() -> SparkSession:
    """
    Initialize PySpark session with the required Kafka SQL package to read streams.
    """
    # Dynamically fetch your exact PySpark version to prevent JAR mismatch
    spark_version = pyspark.__version__
    
    # PySpark 3.x uses Scala 2.12 by default!
    kafka_pkg = f"org.apache.spark:spark-sql-kafka-0-10_2.12:{spark_version}"
    
    print(f"🚨 Booting Spark {spark_version} with Kafka Package: {kafka_pkg} 🚨")
    
    return SparkSession.builder \
        .appName("ARES_Spark_Stream_Processor") \
        .config("spark.jars.packages", kafka_pkg) \
        .getOrCreate()

def main():
    spark = create_spark_session()
    
    # Reduce logging verbosity
    spark.sparkContext.setLogLevel("WARN")
    
    kafka_broker = "localhost:9092"
    topic = "ecommerce-events"
    
    print(f"Setting up Kafka consumer loop. Broker: {kafka_broker}, Topic: {topic}")
    
    # 1. Read from Kafka
    raw_stream = spark \
        .readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", kafka_broker) \
        .option("subscribe", topic) \
        .option("startingOffsets", "earliest") \
        .load()
        
    # 2. Define schema corresponding to the simulated JSON payload format
    schema = StructType([
        StructField("user_id", IntegerType(), True),
        StructField("product_id", IntegerType(), True),
        StructField("price", DoubleType(), True),
        StructField("event_type", StringType(), True),
        StructField("is_fraud", IntegerType(), True)
    ])
    
    # 3. Parse JSON values and extract columns
    # Kafka 'value' column is binary, so we first cast to STRING, then parse as JSON using schema
    parsed_stream = raw_stream \
        .selectExpr("CAST(value AS STRING)") \
        .select(from_json(col("value"), schema).alias("data")) \
        .select("data.*")
        
    # 4. Apply real-time transformation: create 'is_high_value' tag
    transformed_stream = parsed_stream \
        .withColumn("is_high_value", when(col("price") > 500, True).otherwise(False))
        
    print("Starting the stream, waiting for data (sending to Inference API)...")
    
    # 5. Output to FastAPI using foreachBatch
    import json
    import requests

    def send_to_inference_api(df, batch_id):
        # Convert the micro-batch DataFrame into a list of JSON strings
        records = df.toJSON().collect()
        
        for record in records:
            data = json.loads(record)
            try:
                # Post to our local FastAPI service
                response = requests.post("http://localhost:8000/predict", json=data, timeout=5)
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                print(f"⚠️ Error sending record to Inference API: {e}")

    query = transformed_stream \
        .writeStream \
        .foreachBatch(send_to_inference_api) \
        .outputMode("append") \
        .option("checkpointLocation", "/tmp/ares_fresh_chkpt_v3") \
        .start()
        
    # Await termination prevents the script from stopping immediately
    query.awaitTermination()

if __name__ == "__main__":
    main()