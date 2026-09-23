# Apache Spark

Apache Spark is a distributed computing engine designed for processing large datasets. A Spark application can run across multiple machines and execute tasks in parallel.

## Spark Architecture

A Spark application typically consists of a driver process and executor processes. The driver coordinates the application and schedules work, while executors run tasks and store data during computation.

## Spark Executors

Executors are processes responsible for executing tasks assigned by the driver. Executors can store data in memory and on disk during computation. The amount of memory available to an executor affects the amount of data and workload it can handle.

## Executor Out-of-Memory Errors

An executor can encounter an out-of-memory error when the workload requires more memory than the executor has available. Large workloads, cached data, or memory-intensive operations can increase memory requirements.

## Spark SQL

Spark SQL allows users to work with structured data using SQL queries. It provides interfaces for querying structured datasets and can be used together with Spark's DataFrame APIs.

## Spark Shuffle

Shuffle operations redistribute data across executors. Operations such as joins and aggregations can require data to move between executors during a shuffle.