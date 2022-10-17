{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "collapsed_sections": [],
      "mount_file_id": "1B70w1x2l-yj_D7qHdSwVrsIpIajGcWf3",
      "authorship_tag": "ABX9TyMrErWF7i4hSUy33GzlTIx1",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/mosama1994/Diabetes-Detection-using-Machine-Learning/blob/main/CS_777_Assignment_4.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "id": "3sJn3MF8hiVi"
      },
      "outputs": [],
      "source": [
        "!apt-get install openjdk-8-jdk-headless -qq > /dev/null\n",
        "!wget -q https://archive.apache.org/dist/spark/spark-3.3.0/spark-3.3.0-bin-hadoop3.tgz\n",
        "!tar xf spark-3.3.0-bin-hadoop3.tgz\n",
        "\n",
        "import os\n",
        "os.environ[\"JAVA_HOME\"] = \"/usr/lib/jvm/java-8-openjdk-amd64\"\n",
        "os.environ[\"SPARK_HOME\"] = \"/content/spark-3.3.0-bin-hadoop3\"\n",
        "\n",
        "!pip install -q findspark\n",
        "import findspark\n",
        "findspark.init()"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from pyspark.sql import SparkSession, SQLContext\n",
        "from pyspark import SparkContext\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "from numpy.linalg import norm"
      ],
      "metadata": {
        "id": "K_p3v94_hpjX"
      },
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "sc = SparkContext()"
      ],
      "metadata": {
        "id": "vSef9dmUiSro"
      },
      "execution_count": 3,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "sql_c = SQLContext(sc)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 0
        },
        "id": "syWRssJQ6Api",
        "outputId": "4029b0cc-5e11-4f38-f237-057a32f22944"
      },
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/content/spark-3.3.0-bin-hadoop3/python/pyspark/sql/context.py:114: FutureWarning: Deprecated in 3.0.0. Use SparkSession.builder.getOrCreate() instead.\n",
            "  FutureWarning,\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "text = sc.textFile('/content/drive/MyDrive/Colab_Notebooks/WikipediaPagesOneDocPerLine1000LinesSmall.txt')"
      ],
      "metadata": {
        "id": "buqlXOCZiuz4"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "list_top = text.flatMap(lambda x: x.split()).map(lambda x: (x,1)).reduceByKey(lambda x,y: x + y).top(20000, lambda x: x[1])"
      ],
      "metadata": {
        "id": "tkc6gNyVi7si"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "list_top_2 = np.sort([x[0] for x in list_top])"
      ],
      "metadata": {
        "id": "Y9-67ckDob5X"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "list_top"
      ],
      "metadata": {
        "id": "ctP4kjMyJ27l"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "list_top_2[18653]"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 36
        },
        "id": "Rg5dmmo-QKDb",
        "outputId": "68b4e5af-62ac-42f3-88b9-15bd14c32849"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "'the'"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "string"
            }
          },
          "metadata": {},
          "execution_count": 83
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "list_top_2"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Dmom8H_6Phly",
        "outputId": "62a940cf-804d-4c58-87f8-cc0e6b801f49"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array(['\"', '\"\"', '\"\",', ..., '…', '→', '−'], dtype='<U21')"
            ]
          },
          "metadata": {},
          "execution_count": 80
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "def create_matrix(doc):\n",
        "  count = np.zeros(20000, dtype=int)\n",
        "  words, counts = np.unique(doc, return_counts=True)\n",
        "  loc_doc = np.isin(words, list_top_2, assume_unique=True)\n",
        "  loc_list = np.isin(list_top_2, words, assume_unique=True)\n",
        "  count[loc_list] = counts[loc_doc]\n",
        "  return (doc[1],[count,len(doc)])"
      ],
      "metadata": {
        "id": "3oguTmzcECHV"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "tf = text.map(lambda x: x.split()).map(create_matrix).mapValues(lambda x: x[0] / x[1]).collect()"
      ],
      "metadata": {
        "id": "2OTxamQALz9f"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def create_matrix_idf(doc):\n",
        "  words = np.unique(doc)\n",
        "  loc = np.isin(list_top_2, words)\n",
        "  exists = np.array([1 if item == True else 0 for item in loc])\n",
        "  return (1, [exists, 1])"
      ],
      "metadata": {
        "id": "jRaFe719-S2O"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "idf = text.map(lambda x: x.split()).map(create_matrix_idf).reduceByKey(lambda x,y: (x[0]+y[0],x[1]+y[1])).mapValues(lambda x: np.log(x[1] / (x[0]+1))).collect()"
      ],
      "metadata": {
        "id": "ANoM5xeVImso"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "idf"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "57RU6JKnTFy2",
        "outputId": "98c3735e-df58-4774-8dfc-ef0fdac6e459"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "[(1, array([4.07454193, 3.44201938, 3.61191841, ..., 4.82831374, 5.29831737,\n",
              "         5.11599581]))]"
            ]
          },
          "metadata": {},
          "execution_count": 119
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "tf_values = np.array([i[1] for i in tf])\n",
        "idf_values = np.array([i[1] for i in idf]).reshape(1,-1)"
      ],
      "metadata": {
        "id": "_x2lWPTEhyT6"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "idf_values.shape"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "xHxYJBWXTOgd",
        "outputId": "8927d2ce-feec-4318-c64d-3d9f4b92ec61"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(1, 20000)"
            ]
          },
          "metadata": {},
          "execution_count": 101
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "tf_idf = tf_values * idf_values"
      ],
      "metadata": {
        "id": "TT13v_Duiu4s"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "tf_idf"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "nK0oDM6mT2kG",
        "outputId": "1b6a2c8c-6632-4e9a-82b5-9854504b2b79"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[0., 0., 0., ..., 0., 0., 0.],\n",
              "       [0., 0., 0., ..., 0., 0., 0.],\n",
              "       [0., 0., 0., ..., 0., 0., 0.],\n",
              "       ...,\n",
              "       [0., 0., 0., ..., 0., 0., 0.],\n",
              "       [0., 0., 0., ..., 0., 0., 0.],\n",
              "       [0., 0., 0., ..., 0., 0., 0.]])"
            ]
          },
          "metadata": {},
          "execution_count": 121
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pd.DataFrame(result, columns=['Key', 'Value']).to_csv('/idf.csv', index=False)"
      ],
      "metadata": {
        "id": "6-AsnFCvOWM4"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "pd.DataFrame(tf_idf, columns=list_top_2).to_csv('/content/drive/MyDrive/Colab_Notebooks/tf_idf.csv', index = False)"
      ],
      "metadata": {
        "id": "m3mr_dXTXHpS"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Data Frame"
      ],
      "metadata": {
        "id": "DXEXQNPK6LjP"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from pyspark.sql.functions import regexp_extract, explode, split, regexp_replace, lower, lit, collect_list, col, udf, length, sum, array, posexplode, log10"
      ],
      "metadata": {
        "id": "bgsUDOJGF9ac"
      },
      "execution_count": 5,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from pyspark.sql.types import IntegerType, ArrayType, FloatType"
      ],
      "metadata": {
        "id": "37nTVAzrYkyl"
      },
      "execution_count": 6,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df = sql_c.read.csv('/content/drive/MyDrive/Colab_Notebooks/WikipediaPagesOneDocPerLine1000LinesSmall.txt',header=False, sep='\\n')"
      ],
      "metadata": {
        "id": "HcV5hf5i6OdH"
      },
      "execution_count": 7,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df_2 = df.withColumn('url',regexp_extract(df._c0,'<(.*?)>',0))"
      ],
      "metadata": {
        "id": "SQRjpWrGr4Lp"
      },
      "execution_count": 8,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df_2 = df_2.withColumn('text',regexp_extract(df_2._c0,'>(.*?)$',0))"
      ],
      "metadata": {
        "id": "qkgYTZinsTgo"
      },
      "execution_count": 9,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df_2 = df_2.withColumn('clean', regexp_replace(df_2.text,r'[^A-Za-z0-9\\s]+',''))"
      ],
      "metadata": {
        "id": "GW47Cek5mbNK"
      },
      "execution_count": 10,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df_2 = df_2.withColumn('clean', lower(df_2.clean))"
      ],
      "metadata": {
        "id": "plGRcK7nnB-c"
      },
      "execution_count": 11,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df_2 = df_2.drop('_c0')\n",
        "df_2 = df_2.drop('text')"
      ],
      "metadata": {
        "id": "x3HLOxhFq1Rl"
      },
      "execution_count": 12,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df_2 = df_2.withColumn('text_array', split('clean','\\s'))"
      ],
      "metadata": {
        "id": "jmMB94LqnOQF"
      },
      "execution_count": 13,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df_2 = df_2.drop('clean')"
      ],
      "metadata": {
        "id": "pGyPcZOnr0UW"
      },
      "execution_count": 14,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df_2 = df_2.withColumn('split_text', explode('text_array'))"
      ],
      "metadata": {
        "id": "y2jW8foEnvQ1"
      },
      "execution_count": 15,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df_text = df_2.select('split_text')"
      ],
      "metadata": {
        "id": "vgG2V8hgs3ev"
      },
      "execution_count": 16,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df_text = df_text.groupBy(df_text.split_text).count()"
      ],
      "metadata": {
        "id": "RfncqLJ1oE3l"
      },
      "execution_count": 17,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df_top_20 = df_text.orderBy(['count'], ascending = [False]).limit(20000)"
      ],
      "metadata": {
        "id": "j60qhjDVjiXz"
      },
      "execution_count": 18,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "lis = df_top_20.select('split_text').collect()"
      ],
      "metadata": {
        "id": "wT3fRHtekukk"
      },
      "execution_count": 19,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "top_20_list = [i[0] for i in lis]"
      ],
      "metadata": {
        "id": "trJFAMabu8Fv"
      },
      "execution_count": 20,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df_2 = df_2.drop('text_array')"
      ],
      "metadata": {
        "id": "oCOLzLAc77JV"
      },
      "execution_count": 21,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df_2 = df_2.groupBy('url','split_text').count()"
      ],
      "metadata": {
        "id": "Y-EEvZO-ATAt"
      },
      "execution_count": 22,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df_2 = df_2.groupBy('url').agg(collect_list(col('split_text')).alias('text_array'),collect_list(col('count')).alias('count_array'),sum(col('count')).alias('total_count'))"
      ],
      "metadata": {
        "id": "x6aWm2n6BC9F"
      },
      "execution_count": 23,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def tf(text_array, count_array, total_count):\n",
        "  counts = [0.0] * 20000\n",
        "  tf = [k/total_count for k in count_array]\n",
        "  for i in range(len(text_array)):\n",
        "    if text_array[i] in top_20_list:\n",
        "      ind = top_20_list.index(text_array[i])\n",
        "      counts[ind] = tf[i]\n",
        "  return counts"
      ],
      "metadata": {
        "id": "_kFTlNHKD1zn"
      },
      "execution_count": 24,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "udf_tf = udf(tf,ArrayType(FloatType()))"
      ],
      "metadata": {
        "id": "nuwi2OSCEhWf"
      },
      "execution_count": 25,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df_2 = df_2.withColumn('tf', udf_tf(col('text_array'),col('count_array'),col('total_count')))"
      ],
      "metadata": {
        "id": "OKSXOyHHBf-u"
      },
      "execution_count": 26,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# IDF Calculation"
      ],
      "metadata": {
        "id": "pej8rNvvyqcZ"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def idf(text_array):\n",
        "  counts = [0] * 20000\n",
        "  for i in range(len(text_array)):\n",
        "    if text_array[i] in top_20_list:\n",
        "      ind = top_20_list.index(text_array[i])\n",
        "      counts[ind] = 1\n",
        "  return counts"
      ],
      "metadata": {
        "id": "SMTYwx5Vyu8K"
      },
      "execution_count": 27,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "udf_idf = udf(idf,ArrayType(IntegerType()))"
      ],
      "metadata": {
        "id": "M94O7e8pMqHR"
      },
      "execution_count": 28,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df_2 = df_2.withColumn('calc_idf', udf_idf(col('text_array')))"
      ],
      "metadata": {
        "id": "w7cLUe0WMw1R"
      },
      "execution_count": 29,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df_idf_calc = df_2.select(posexplode('calc_idf'))"
      ],
      "metadata": {
        "id": "HgkYh-TTRWwn"
      },
      "execution_count": 30,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df_idf_calc = df_idf_calc.groupBy('pos').sum('col')"
      ],
      "metadata": {
        "id": "taAds3vicXZX"
      },
      "execution_count": 31,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df_idf_calc = df_idf_calc.withColumn('idf', log10(df.count()/col('sum(col)'))).orderBy('pos')"
      ],
      "metadata": {
        "id": "nTg9sPXIc9iE"
      },
      "execution_count": 32,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "list_idf = df_idf_calc.select('idf').collect()"
      ],
      "metadata": {
        "id": "5uXBCLggfHon"
      },
      "execution_count": 33,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "list_idf = [i[0] for i in list_idf]"
      ],
      "metadata": {
        "id": "uXKxz23kozWG"
      },
      "execution_count": 34,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def multiply(a, b):\n",
        "  return [x * y for x,y in zip(a,b)]"
      ],
      "metadata": {
        "id": "To0vP9wRl9_u"
      },
      "execution_count": 35,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "udf_multiply = udf(lambda x: multiply(list_idf,x), ArrayType(FloatType()))"
      ],
      "metadata": {
        "id": "YvU3WFyVmLoW"
      },
      "execution_count": 36,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df_2 = df_2.withColumn('tf_idf', udf_multiply(df_2['tf']))"
      ],
      "metadata": {
        "id": "jhcFrcH6glZN"
      },
      "execution_count": 37,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df_2 = df_2.withColumn('doc_id', regexp_extract(col('url'), r'doc id=\"(.*?)\"',0))"
      ],
      "metadata": {
        "id": "LWuWjEVYq0z3"
      },
      "execution_count": 38,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df_3 = df_2.select('doc_id','tf_idf')"
      ],
      "metadata": {
        "id": "wnyIFwUDvsCZ"
      },
      "execution_count": 39,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df_3.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 0
        },
        "id": "Fo3GT30Svxqg",
        "outputId": "714db5c7-8380-4985-db23-0e23675b45e8"
      },
      "execution_count": 40,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "+---------------+--------------------+\n",
            "|         doc_id|              tf_idf|\n",
            "+---------------+--------------------+\n",
            "|doc id=\"418293\"|[0.004044701, 0.0...|\n",
            "|doc id=\"418300\"|[0.0028331093, 0....|\n",
            "|doc id=\"418309\"|[0.0025657716, 0....|\n",
            "|doc id=\"418311\"|[0.0024696023, 0....|\n",
            "|doc id=\"418319\"|[0.0061928863, 0....|\n",
            "|doc id=\"418321\"|[0.0021441302, 0....|\n",
            "|doc id=\"418322\"|[0.003744536, 7.7...|\n",
            "|doc id=\"418329\"|[0.0024915987, 0....|\n",
            "|doc id=\"418332\"|[0.0030332506, 0....|\n",
            "|doc id=\"418338\"|[0.0028037315, 0....|\n",
            "|doc id=\"418347\"|[0.0029937176, 6....|\n",
            "|doc id=\"418351\"|[0.0016070148, 8....|\n",
            "|doc id=\"418352\"|[0.0020642956, 0....|\n",
            "|doc id=\"418355\"|[0.0017861759, 0....|\n",
            "|doc id=\"418359\"|[0.0026877755, 5....|\n",
            "|doc id=\"418362\"|[0.00316779, 9.17...|\n",
            "|doc id=\"418370\"|[0.0, 0.0, 0.0, 0...|\n",
            "|doc id=\"418375\"|[0.0029663406, 0....|\n",
            "|doc id=\"418376\"|[0.0031973298, 0....|\n",
            "|doc id=\"418380\"|[0.0025305778, 8....|\n",
            "+---------------+--------------------+\n",
            "only showing top 20 rows\n",
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# KNN"
      ],
      "metadata": {
        "id": "UH7E8SHuv4xy"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Selecting doc id ='418359' as the input\n",
        "input_doc = df_3.filter(df_3.doc_id == 'doc id=\"418359\"').select('tf_idf').collect()"
      ],
      "metadata": {
        "id": "1Tv8Up1Tv4Kp"
      },
      "execution_count": 41,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "input_doc = [i[0] for i in input_doc][0]"
      ],
      "metadata": {
        "id": "O_5KxWUCzdWK"
      },
      "execution_count": 42,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def cosine(a, b):\n",
        "  if (norm(a) * norm(b)) == 0: return float(99.0)\n",
        "  else: return float(np.dot(a,b)/(norm(a)*norm(b)))"
      ],
      "metadata": {
        "id": "3WHIs0Py3-w0"
      },
      "execution_count": 43,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "udf_cosine = udf(lambda x: (cosine(x,input_doc)), FloatType())"
      ],
      "metadata": {
        "id": "8sXKa9Rv3-y7"
      },
      "execution_count": 44,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df_3 = df_3.withColumn('cosine', udf_cosine(col('tf_idf')))"
      ],
      "metadata": {
        "id": "xg1CnHBHzhwC"
      },
      "execution_count": 45,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df_3 = df_3.filter(col('cosine') != 99.0)"
      ],
      "metadata": {
        "id": "ugqiCEqbYGJN"
      },
      "execution_count": 48,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df_3 = df_3.orderBy(['cosine'], ascending = [False]).limit(20)"
      ],
      "metadata": {
        "id": "mHQALtf67N_9"
      },
      "execution_count": 49,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df_3.select('doc_id','cosine').write.mode(\"overwrite\").options(header=True).csv('/content/drive/MyDrive/CS_777_Assignments/top_20_doc')"
      ],
      "metadata": {
        "id": "HABsGX2Y6sNl"
      },
      "execution_count": 50,
      "outputs": []
    }
  ]
}