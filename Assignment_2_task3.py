from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import *
import sys
from pyspark.sql.functions import lit
from pyspark.sql import functions as func

if __name__ == "__main__":
    wikiPagesFile = sys.argv[1]
    wikiCategoryFile = sys.argv[2]
    output_dir_1 = sys.argv[3]
    output_dir_2 = sys.argv[4]
    output_dir_3 = sys.argv[5]

    # wikiPagesFile = "WikipediaPages_oneDocPerLine_1000Lines_small.txt"
    # wikiCategoryFile = "wiki-categorylinks-small.csv.bz2"
    # output_dir_1 = 're_1'
    # output_dir_2 = 're_2'
    # output_dir_3 = 're_3'


    sc = SparkContext()
    # Now the wikipages
    wikiPages = sc.textFile(wikiPagesFile)

    sql_context = SQLContext(sc)

    validLines = wikiPages.filter(lambda x: 'id' in x and 'url=' in x)

    # Now, we transform it into a set of (docID, text) pairs
    keyAndText = validLines.map(lambda x: (x[x.index('id="') + 4: x.index('" url=')], x[x.index('">') + 2:][:-6]))

    schema = StructType([
            StructField("ID", StringType(), False),
            StructField("Category", StringType(), False),
            ]
        )

    cats_df = sql_context.read.csv(wikiCategoryFile, header=False, schema=schema)
    page_df = sql_context.createDataFrame(keyAndText, ["page_id", "Text"])


    df = cats_df.join(page_df, cats_df.ID == page_df.page_id, "inner")

    df = df.withColumn("COUNT", lit(1))
    cat_sum_df = df.groupBy('Category').agg(func.sum("COUNT")).withColumnRenamed("sum(COUNT)", 'cat_count')
    # Task 3.1
    summary = cat_sum_df.describe('cat_count')
    summary.show()
    re_1 = summary.rdd.coalesce(1)
    re_1.saveAsTextFile(output_dir_1)

    top_cat_df = cat_sum_df.orderBy(cat_sum_df.cat_count.desc()).limit(10)
    # Task 3.2
    top_cat_df.show()
    re_2 = top_cat_df.rdd.coalesce(1)
    re_2.saveAsTextFile(output_dir_2)

    page_with_top_cat = df.filter(top_cat_df.Category == df.Category)
    page_sum_df = page_with_top_cat.groupBy("ID").agg(func.sum("COUNT")).withColumnRenamed("sum(COUNT)",'page_count')
    # Task 3.3
    page_top_df = page_sum_df.orderBy(page_sum_df.page_count.desc()).limit(10)
    page_top_df.show()
    re_3 = page_top_df.rdd.coalesce(1)
    re_3.saveAsTextFile(output_dir_3)
